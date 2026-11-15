"""Statistics and figures for the technical report, from out-of-fold predictions.

usage: python tools/report_stats.py --probs DIR[,DIR2] --cfg "thr=0.5;seed_thr=0;min_area=0" \
           [--rescorer r.pkl --score_thr 0.3] --out DIR

Writes (into --out):
  stats.json        PQ / SQ / RQ with 95% bootstrap intervals (resampling images), per-observatory PQ,
                    one-to-many / many-to-one counts (pairs with IoU > 0), IoU and Dice of TP pairs
  iou_dice.png      distributions of IoU and Dice of overlapping GT/prediction pairs
  examples.png      a few out-of-fold images: H-alpha, annotator masks, predicted instances
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filseg.metric import GTCache, pq_from_iou  # noqa: E402
from filseg.postproc import instances, upsample  # noqa: E402
from filseg.rescore import component_features  # noqa: E402
from eval_probs import GT_CSV, load_prob, parse_grid  # noqa: E402

from filseg.paths import TRAIN_IMAGES as TRAIN_DIR  # noqa: E402  ($FILAMENT_DATA/train/train_images)


def segment(prob, cfg, resc, thr):
    lab, n = instances(prob, cfg)
    if resc is not None and n:
        keep = resc.predict(component_features(prob, lab, n)) >= thr
        remap = np.zeros(n + 1, np.int32)
        remap[1:][keep] = np.arange(1, keep.sum() + 1)
        lab, n = remap[lab], int(keep.sum())
    return lab, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--rescorer", default="")
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = a.probs.split(",")
    cfg = parse_grid(a.cfg)[0]
    resc = pickle.load(open(a.rescorer, "rb"))["model"] if a.rescorer else None
    imgs = sorted(set.intersection(*[{f[:-4] for f in os.listdir(d) if f.endswith(".png")} for d in dirs]))
    gt = pd.read_csv(GT_CSV)
    gt = gt[gt.image.isin(imgs)]
    cache = GTCache(gt.to_dict("records"))
    per_img, ious, dices, one_many, many_one, obs = [], [], [], [], [], []
    keep_lab = {}
    ex = a.examples.split(",") if a.examples else imgs[::max(1, len(imgs) // 3)][:3]
    for img in imgs:
        prob = upsample(load_prob(dirs, img))
        lab, n = segment(prob, cfg, resc, a.score_thr)
        if img in ex:
            keep_lab[img] = lab
        acc = np.zeros(4)
        for ai, m in cache.iou_matrices(img, lab, n).items():
            s = pq_from_iou(m, m.shape[0], n)
            acc += [s.tp_iou, s.tp, s.fp, s.fn]
            if m.size:
                ov = m > 0
                one_many += ov.sum(1).tolist()   # predictions touching each GT filament
                many_one += ov.sum(0).tolist()   # GT filaments touching each prediction
                hit = m > 0.5
                ious += m[hit].tolist()
                dices += (2 * m[hit] / (1 + m[hit])).tolist()
        per_img.append(acc)
        obs.append(img[14:16])
    A = np.array(per_img)

    def pq(x):
        s = x.sum(0)
        d = s[1] + 0.5 * s[2] + 0.5 * s[3]
        return s[0] / d, s[0] / max(s[1], 1), s[1] / d

    rng = np.random.RandomState(0)
    boot = np.array([pq(A[rng.randint(0, len(A), len(A))]) for _ in range(2000)])
    PQ, SQ, RQ = pq(A)
    obs = np.array(obs)
    stats = {
        "images": len(imgs), "PQ": PQ, "SQ": SQ, "RQ": RQ,
        "PQ_ci95": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
        "SQ_ci95": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
        "RQ_ci95": np.percentile(boot[:, 2], [2.5, 97.5]).tolist(),
        "TP_FP_FN": A[:, 1:].sum(0).astype(int).tolist(),
        "per_observatory_PQ": {o: pq(A[obs == o])[0] for o in sorted(set(obs))},
        "tp_iou_mean": float(np.mean(ious)), "tp_dice_mean": float(np.mean(dices)),
        "gt_touched_by_k_preds": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(one_many, 4), return_counts=True))},
        "pred_touching_k_gts": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(many_one, 4), return_counts=True))},
    }
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(ious, bins=25, range=(0.5, 1), color="#3b6ea5")
    ax[0].set_xlabel("IoU of matched pairs")
    ax[1].hist(dices, bins=25, range=(0.66, 1), color="#a5603b")
    ax[1].set_xlabel("Dice of matched pairs")
    for x in ax:
        x.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "iou_dice.png"), dpi=200)

    fig, axes = plt.subplots(len(keep_lab), 3, figsize=(9, 3 * len(keep_lab)))
    axes = np.atleast_2d(axes)
    for row, (img, lab) in zip(axes, keep_lab.items()):
        im = cv2.imread(os.path.join(TRAIN_DIR, img + ".jpeg"), cv2.IMREAD_GRAYSCALE)
        g = gt[gt.image == img]
        first = g.annotator_image.iloc[0]
        from filseg.data import counts_to_mask
        gl = np.zeros(im.shape, np.int32)
        for k, c in enumerate(g[g.annotator_image == first].segmentation_rle, 1):
            gl[counts_to_mask(c) > 0] = k
        row[0].imshow(im, cmap="gray"); row[0].set_title(img, fontsize=8)
        row[1].imshow(im, cmap="gray"); row[1].imshow(np.ma.masked_equal(gl, 0) % 20, cmap="tab20", alpha=0.7, interpolation="nearest")
        row[1].set_title(f"annotator {first.split('-')[0]}", fontsize=8)
        row[2].imshow(im, cmap="gray"); row[2].imshow(np.ma.masked_equal(lab, 0) % 20, cmap="tab20", alpha=0.7, interpolation="nearest")
        row[2].set_title("prediction (out-of-fold)", fontsize=8)
        for x in row:
            x.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "examples.png"), dpi=150)
    print("figures ->", a.out)
