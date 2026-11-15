"""Train a U-Net filament segmenter on MAGFiLO (Kaggle GPU kernel).

Outputs (in /kaggle/working):
  folds.csv                       image -> fold (GroupKFold-like split by observation date)
  model_f{k}.pth                  fp16 state_dict of the fold-k model
  oof/{image}.png                 out-of-fold foreground probability (uint8, CFG.size x CFG.size)
  test/f{k}/{image}.png           test probability of the fold-k model (uint8)
  log.txt                         training log

The probability maps are post-processed (threshold, connected components, filtering) and
scored with the exact organisers' PQ outside this kernel (see src/filseg).
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

CFG = dict(
    encoder=os.environ.get("ENC", "tu-tf_efficientnetv2_s"),
    size=int(os.environ.get("SIZE", "1536")),
    epochs=int(os.environ.get("EPOCHS", "25")),
    batch=int(os.environ.get("BATCH", "4")),
    lr=float(os.environ.get("LR", 5e-4)),
    folds=[int(f) for f in os.environ.get("FOLDS", "0").split(",")],
    nfolds=5,
    seed=42,
    full_train=int(os.environ.get("FULL", 0)),  # 1 = also train one model on all images
    crop=int(os.environ.get("CROP", 0)),          # 0 = train on full images, else random crops of this size
    crops_per_img=int(os.environ.get("CPI", 1)),  # crops drawn per image per epoch
    arch=os.environ.get("ARCH", "Unet"),
)
OUT = "/kaggle/working"
T0 = time.time()


def log(*a):
    msg = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(f"{OUT}/log.txt", "a") as f:
        f.write(msg + "\n")


def ensure(pkg, pip_name=None):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg])


ensure("segmentation_models_pytorch", "segmentation-models-pytorch")
ensure("pycocotools")
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from pycocotools import mask as mask_util

# ----------------------------------------------------------------------------- data
ann_path = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)[0]
root = os.path.dirname(os.path.dirname(ann_path))
train_dir = os.path.join(root, "train", "train_images")
test_dir = os.path.join(root, "test", "test_images")
log("data root", root, CFG)

coco = json.load(open(ann_path))
anns_by_img = defaultdict(list)
for a in coco["annotations"]:
    anns_by_img[a["image_id"]].append(a)
annotators = defaultdict(list)  # image name -> list of annotator-image ids
for im in coco["images"]:
    annotators[im["file_name"].rsplit(".", 1)[0]].append(im["id"])
train_names = sorted(annotators)
test_names = sorted(f.rsplit(".", 1)[0] for f in os.listdir(test_dir))
S = CFG["size"]


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)


def soft_target(name: str) -> np.ndarray:
    """Mean over annotators of the union mask, rasterised with pycocotools at 2048 then area-resized."""
    acc = np.zeros((S, S), np.float32)
    for aid in annotators[name]:
        u = np.zeros((2048, 2048), np.uint8)
        for a in anns_by_img[aid]:
            rle = mask_util.merge(mask_util.frPyObjects(a["segmentation"], 2048, 2048))
            u |= mask_util.decode(rle)
        acc += cv2.resize(u.astype(np.float32), (S, S), interpolation=cv2.INTER_AREA)
    return acc / len(annotators[name])


def disk_stats(img: np.ndarray) -> tuple[float, float]:
    """Robust location/scale of the solar disk (pixels > 20): median and IQR/1.349."""
    x = img[img > 20].astype(np.float32)
    if x.size < 1000:
        x = img.astype(np.float32).ravel()
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(max((q3 - q1) / 1.349, 1.0))


def normalise_t(x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    """x: (B,1,H,W) raw 0..255 floats, stats: (B,2) -> per-image standardised disk, off-disk = -2."""
    med = stats[:, 0].view(-1, 1, 1, 1)
    sc = stats[:, 1].view(-1, 1, 1, 1)
    z = ((x - med) / sc).clamp(-8, 8)
    z = torch.where(x > 20, z, torch.full_like(z, -8))
    return z / 4.0


log("loading", len(train_names), "train and", len(test_names), "test images")
X = np.stack([load_image(os.path.join(train_dir, n + ".jpeg")) for n in train_names])
Y = np.stack([np.round(soft_target(n) * 255).astype(np.uint8) for n in train_names])
XT = np.stack([load_image(os.path.join(test_dir, n + ".jpeg")) for n in test_names])
ST = torch.tensor([disk_stats(x) for x in X], dtype=torch.float32)
STT = torch.tensor([disk_stats(x) for x in XT], dtype=torch.float32)
log("loaded", X.shape, Y.shape, XT.shape, "fg frac", float(Y.mean() / 255))

# split: group by observation date so near-duplicate frames of one day never straddle folds
dates = sorted({n[:8] for n in train_names})
rng = np.random.RandomState(CFG["seed"])
rng.shuffle(dates)
date_fold = {d: i % CFG["nfolds"] for i, d in enumerate(dates)}
fold_of = np.array([date_fold[n[:8]] for n in train_names])
with open(f"{OUT}/folds.csv", "w") as f:
    f.write("image,fold\n")
    for n, k in zip(train_names, fold_of):
        f.write(f"{n},{k}\n")

# ----------------------------------------------------------------------------- model / training
dev = "cuda"
ngpu = torch.cuda.device_count()
log("gpus", ngpu, torch.cuda.get_device_name(0) if ngpu else "-")


def build_model():
    return getattr(smp, CFG["arch"])(CFG["encoder"], encoder_weights="imagenet", in_channels=1, classes=1)


def augment(x: torch.Tensor, y: torch.Tensor):
    """GPU batch augmentation: dihedral flips, small rotation+scale, intensity jitter."""
    b = x.shape[0]
    if random.random() < 0.5:
        x, y = x.flip(3), y.flip(3)
    if random.random() < 0.5:
        x, y = x.flip(2), y.flip(2)
    k = random.randint(0, 3)
    x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
    ang = (torch.rand(b, device=x.device) - 0.5) * 2 * math.pi / 12  # +-15 deg
    sc = 1 + (torch.rand(b, device=x.device) - 0.5) * 0.2           # 0.9..1.1
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                         torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y = F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # intensity: contrast / brightness / gamma-like jitter, plus mild noise
    c = 1 + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.4
    br = (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.2
    x = x * c + br + torch.randn_like(x) * 0.02
    return x, y


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def predict(net, arr: np.ndarray, stats: torch.Tensor, bs: int = 2) -> np.ndarray:
    """Flip-TTA (4 flips) probability maps of full images, returned as uint8 (N,S,S)."""
    net.eval()
    out = np.zeros(arr.shape, np.uint8)
    for i in range(0, len(arr), bs):
        x = normalise_t(torch.from_numpy(arr[i:i + bs]).float()[:, None].to(dev), stats[i:i + bs].to(dev))
        acc = 0
        with torch.autocast("cuda", dtype=torch.float16):
            for fh in (False, True):
                for fv in (False, True):
                    xx = x.flip(3) if fh else x
                    xx = xx.flip(2) if fv else xx
                    p = torch.sigmoid(net(xx).float())
                    p = p.flip(2) if fv else p
                    p = p.flip(3) if fh else p
                    acc = acc + p
        out[i:i + bs] = (acc / 4 * 255 + 0.5).clamp(0, 255)[:, 0].byte().cpu().numpy()
    return out


def save_probs(probs: np.ndarray, names, folder: str):
    os.makedirs(folder, exist_ok=True)
    for p, n in zip(probs, names):
        cv2.imwrite(os.path.join(folder, n + ".png"), p)


FG_PTS = {}


def crop_centre(j: int) -> tuple[int, int]:
    """Half of the crops are centred on a random annotated pixel, half on a random disk position."""
    if j not in FG_PTS:
        ys, xs = np.nonzero(Y[j][::4, ::4] > 0)
        FG_PTS[j] = np.stack([ys, xs], 1) * 4
    pts = FG_PTS[j]
    if len(pts) and random.random() < 0.5:
        cy, cx = pts[random.randrange(len(pts))]
    else:
        r, t = 0.45 * S * math.sqrt(random.random()), random.random() * 2 * math.pi
        cy, cx = S / 2 + r * math.sin(t), S / 2 + r * math.cos(t)
    return int(cy), int(cx)


def train_one(tr_idx, va_idx, tag):
    torch.manual_seed(CFG["seed"])
    random.seed(CFG["seed"])
    model = build_model().to(dev)
    net = torch.nn.DataParallel(model) if ngpu > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(tr_idx) * (CFG["crops_per_img"] if CFG["crop"] else 1) / CFG["batch"])
    total = CFG["epochs"] * steps_per_epoch
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.cuda.amp.GradScaler()
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    C = CFG["crop"]
    for ep in range(CFG["epochs"]):
        net.train()
        perm = np.concatenate([np.random.permutation(tr_idx) for _ in range(CFG["crops_per_img"] if C else 1)])
        tl, nb = 0.0, 0
        for i in range(0, len(perm), CFG["batch"]):
            idx = perm[i:i + CFG["batch"]]
            if len(idx) < 2:
                continue
            if C:
                xs, ys = [], []
                for j in idx:
                    cy, cx = crop_centre(j)
                    y0 = int(np.clip(cy - C // 2, 0, S - C))
                    x0 = int(np.clip(cx - C // 2, 0, S - C))
                    xs.append(Xt[j, y0:y0 + C, x0:x0 + C])
                    ys.append(Yt[j, y0:y0 + C, x0:x0 + C])
                xb, yb = torch.stack(xs), torch.stack(ys)
            else:
                ti = torch.from_numpy(idx)
                xb, yb = Xt[ti], Yt[ti]
            x = normalise_t(xb.float()[:, None].to(dev, non_blocking=True), ST[torch.from_numpy(idx)].to(dev))
            y = yb.float()[:, None].to(dev, non_blocking=True) / 255.0
            x, y = augment(x, y)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
            nb += 1
        msg = f"{tag} ep {ep + 1}/{CFG['epochs']} loss {tl / max(nb, 1):.4f}"
        if len(va_idx) and ((ep + 1) % 5 == 0 or ep + 1 == CFG["epochs"]):
            pv = predict(net, X[va_idx[:40]], ST[torch.from_numpy(va_idx[:40])])
            yv = Y[va_idx[:40]] >= 128
            pb = pv >= 128
            dice = 2 * (pb & yv).sum() / max(pb.sum() + yv.sum(), 1)
            msg += f" | val pixel-dice@0.5 (40 imgs) {dice:.4f}"
        log(msg)
    torch.save({k: v.half() for k, v in model.state_dict().items()}, f"{OUT}/model_{tag}.pth")
    return net


for k in CFG["folds"]:
    tr_idx = np.where(fold_of != k)[0]
    va_idx = np.where(fold_of == k)[0]
    log(f"fold {k}: train {len(tr_idx)} val {len(va_idx)}")
    model = train_one(tr_idx, va_idx, f"f{k}")
    save_probs(predict(model, X[va_idx], ST[torch.from_numpy(va_idx)]), [train_names[i] for i in va_idx], f"{OUT}/oof")
    save_probs(predict(model, XT, STT), test_names, f"{OUT}/test/f{k}")
    del model
    torch.cuda.empty_cache()

if CFG["full_train"]:
    model = train_one(np.arange(len(train_names)), np.array([], int), "full")
    save_probs(predict(model, XT, STT), test_names, f"{OUT}/test/full")

json.dump(CFG, open(f"{OUT}/cfg.json", "w"))
log("done")
