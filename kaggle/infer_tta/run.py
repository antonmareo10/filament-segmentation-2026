"""Inference-only kernel: extra test-time views for the 10 final models (no training).

For every fold model (EffNetV2-S 1536, ResNet-34 1536) it writes, for its out-of-fold images and for the test set:
  <view>/oof/{image}.png and <view>/test/{arch}_f{k}/{image}.png   (uint8 probability, model resolution)
views:
  t1536  : mean of the 4 transposed flips at 1536 px (combined with the stored 4 flips -> 8 dihedral views)
  s2048  : mean of the 4 flips at native 2048 px (multi-scale TTA)
"""
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

T0 = time.time()
OUT = "/kaggle/working"


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


try:
    import segmentation_models_pytorch as smp  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"])
import torch
import segmentation_models_pytorch as smp

ann = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)[0]
root = os.path.dirname(os.path.dirname(ann))
train_dir, test_dir = os.path.join(root, "train", "train_images"), os.path.join(root, "test", "test_images")
coco = json.load(open(ann))
train_names = sorted({im["file_name"].rsplit(".", 1)[0] for im in coco["images"]})
test_names = sorted(f.rsplit(".", 1)[0] for f in os.listdir(test_dir))
dates = sorted({n[:8] for n in train_names})
rng = np.random.RandomState(42)
rng.shuffle(dates)
date_fold = {d: i % 5 for i, d in enumerate(dates)}
fold_of = {n: date_fold[n[:8]] for n in train_names}

weights = {}
for p in glob.glob("/kaggle/input/**/model_f*.pth", recursive=True):
    arch = "effv2s" if "effv2s-1536" in p else ("r34" if "r34-1536" in p else None)
    if arch:
        weights[(arch, int(p.rsplit("_f", 1)[1][0]))] = p
log("weights", sorted(weights))
ENC = {"effv2s": "tu-tf_efficientnetv2_s", "r34": "resnet34"}


def disk_stats(img):
    x = img[img > 20].astype(np.float32)
    if x.size < 1000:
        x = img.astype(np.float32).ravel()
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(max((q3 - q1) / 1.349, 1.0))


def prep(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    med, sc = disk_stats(img)
    x = np.clip((img.astype(np.float32) - med) / sc, -8, 8)
    x[img <= 20] = -8
    return x / 4.0


@torch.no_grad()
def views(net, x, transposed):
    """x: (1,1,H,W). Mean sigmoid over the 4 flips of x (or of x transposed)."""
    if transposed:
        x = x.transpose(2, 3)
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
    acc = acc / 4
    if transposed:
        acc = acc.transpose(2, 3)
    return acc[0, 0]


def save(p, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (p.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy())


def run_model(arch, k, dev):
    net = smp.Unet(ENC[arch], encoder_weights=None, in_channels=1, classes=1)
    net.load_state_dict({a: b.float() for a, b in torch.load(weights[(arch, k)], map_location="cpu").items()})
    net = net.to(dev).eval()
    jobs = [(n, train_dir, f"oof") for n in train_names if fold_of[n] == k] + [(n, test_dir, f"test/{arch}_f{k}") for n in test_names]
    for n, d, sub in jobs:
        path = os.path.join(d, n + ".jpeg")
        x15 = torch.from_numpy(prep(path, 1536))[None, None].to(dev)
        save(views(net, x15, True), f"{OUT}/t1536/{arch}/{sub}/{n}.png")
        x20 = torch.from_numpy(prep(path, 2048))[None, None].to(dev)
        save(views(net, x20, False), f"{OUT}/s2048/{arch}/{sub}/{n}.png")
    log(arch, k, "done", len(jobs))


import threading

todo = sorted(weights)
ngpu = torch.cuda.device_count()


def worker(g):
    for i, (arch, k) in enumerate(todo):
        if i % ngpu == g:
            run_model(arch, k, f"cuda:{g}")


ths = [threading.Thread(target=worker, args=(g,)) for g in range(ngpu)]
for t in ths:
    t.start()
for t in ths:
    t.join()
log("all done")
