"""Zoomed examples for the report: H-alpha crop | every annotator's filaments | our out-of-fold instances.

usage: python tools/fig_examples.py --probs DIR[,DIR2,...] --cfg "thr=0.5;seed_thr=0;min_area=0;group_r=10"
           (--cands c.pkl --folds folds.csv | --rescorer r.pkl) --score_thr 0.325
           --images img1,img2,img3 [--labels "good,typical,hard"] --out report/fig_examples.png

The probability of an image is the mean of its maps in the --probs folders (out-of-fold maps of the training kernels).
With --cands/--folds the keep decision of every candidate is the *cross-fitted* re-scorer score (fitted on the
other folds, as in the reported statistics), so the figure shows exactly what was scored out-of-fold.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filseg.data import counts_to_mask  # noqa: E402
from filseg.metric import GTCache, pq_from_iou  # noqa: E402
from filseg.paths import TRAIN_IMAGES  # noqa: E402
from filseg.postproc import instances, upsample  # noqa: E402
from filseg.rescore import component_features  # noqa: E402
from eval_probs import GT_CSV, load_prob, parse_grid  # noqa: E402

ANNOTATOR_COLORS = [(42, 120, 214), (235, 104, 52), (27, 175, 122)]          # one colour per annotator
INSTANCE_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
                   (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40)]  # one colour per predicted filament


def outlines(img, label_maps, crop, colors, by_map, thickness):
    y0, y1, x0, x1 = crop
    rgb = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2RGB)
    for li, lab in enumerate(label_maps):
        sub = lab[y0:y1, x0:x1]
        for k in np.unique(sub[sub > 0]):
            cs, _ = cv2.findContours((sub == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            col = colors[li % len(colors)] if by_map else colors[int(k) % len(colors)]
            cv2.drawContours(rgb, cs, -1, col, thickness)
    return rgb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--rescorer", default="", help="fitted re-scorer (in-sample for training images)")
    ap.add_argument("--cands", default="", help="candidate file of the same --probs/--cfg for cross-fitted scores")
    ap.add_argument("--folds", default="")
    ap.add_argument("--score_thr", type=float, default=0.325)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--size", type=int, default=560, help="crop size in native pixels")
    ap.add_argument("--width", type=float, default=3.35, help="figure width in inches (one template column)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = parse_grid(a.cfg)[0]
    names = a.images.split(",")
    labels = a.labels.split(",") if a.labels else [""] * len(names)
    scores = None
    if a.cands:
        from report_from_cands import crossfit_scores
        scores = crossfit_scores(pickle.load(open(a.cands, "rb"))["data"], dict(pd.read_csv(a.folds).values))
    resc = pickle.load(open(a.rescorer, "rb"))["model"] if a.rescorer else None
    gt = pd.read_csv(GT_CSV)
    cache = GTCache(gt[gt.image.isin(names)].to_dict("records"))
    plt.rcParams.update({"font.size": 6, "pdf.fonttype": 42})
    pw = a.width / 3
    fig, axes = plt.subplots(len(names), 3, figsize=(a.width, pw * len(names) + 0.12), squeeze=False)
    for row, img, lbl in zip(axes, names, labels):
        im = cv2.imread(os.path.join(TRAIN_IMAGES, img + ".jpeg"), cv2.IMREAD_GRAYSCALE)
        prob = upsample(load_prob(a.probs.split(","), img))
        lab, n = instances(prob, cfg)
        keep = None
        if scores is not None and n:
            keep = scores[img] >= a.score_thr
            assert len(keep) == n, "candidate file does not match --probs/--cfg"
        elif resc is not None and n:
            keep = resc.predict(component_features(prob, lab, n)) >= a.score_thr
        if keep is not None:
            remap = np.zeros(n + 1, np.int32)
            remap[1:][keep] = np.arange(1, keep.sum() + 1)
            lab, n = remap[lab], int(keep.sum())
        acc = np.zeros(4)
        for m in cache.iou_matrices(img, lab, n).values():
            s = pq_from_iou(m, m.shape[0], n)
            acc += [s.tp_iou, s.tp, s.fp, s.fn]
        pq = acc[0] / (acc[1] + 0.5 * acc[2] + 0.5 * acc[3])
        g = gt[gt.image == img]
        ann_maps = []
        for _, ga in g.groupby("annotator_image"):
            m = np.zeros(im.shape, np.int32)
            for k, c in enumerate(ga.segmentation_rle, 1):
                m[counts_to_mask(c) > 0] = k
            ann_maps.append(m)
        # crop around the densest annotated region
        allm = sum((m > 0).astype(np.float32) for m in ann_maps)
        sm = cv2.boxFilter(allm, -1, (a.size, a.size))
        cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
        y0 = int(np.clip(cy - a.size // 2, 0, 2048 - a.size))
        x0 = int(np.clip(cx - a.size // 2, 0, 2048 - a.size))
        crop = (y0, y0 + a.size, x0, x0 + a.size)
        th = max(2, a.size // 160)
        row[0].imshow(im[crop[0]:crop[1], crop[2]:crop[3]], cmap="gray", interpolation="lanczos")
        row[1].imshow(outlines(im, ann_maps, crop, ANNOTATOR_COLORS, True, th), interpolation="lanczos")
        row[2].imshow(outlines(im, [lab], crop, INSTANCE_COLORS, False, th), interpolation="lanczos")
        site = img[14:16]
        row[0].set_ylabel(f"{lbl}\n{img[:8]} {site}", fontsize=5.6, labelpad=2)
        row[1].text(0.02, 0.98, f"{len(ann_maps)} annotators", transform=row[1].transAxes, ha="left", va="top",
                    fontsize=5.4, color="white", bbox=dict(fc="black", alpha=0.45, lw=0, pad=1))
        row[2].text(0.02, 0.98, f"image PQ {pq:.2f}", transform=row[2].transAxes, ha="left", va="top",
                    fontsize=5.4, color="white", bbox=dict(fc="black", alpha=0.45, lw=0, pad=1))
        for z in row:
            z.set_xticks([])
            z.set_yticks([])
            for sp in z.spines.values():
                sp.set_linewidth(0.4)
        print(f"{img}: {len(ann_maps)} annotators, {n} predicted filaments, image PQ {pq:.3f}, crop {crop}")
    for z, t in zip(axes[0], [r"H$\alpha$ crop", "experts (colour = annotator)", "ours, out-of-fold"]):
        z.set_title(t, fontsize=6, pad=2)
    plt.subplots_adjust(left=0.09, right=0.995, top=1 - 0.12 / (pw * len(names) + 0.12), bottom=0.003, wspace=0.03, hspace=0.03)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(a.out, dpi=400)
    plt.savefig(a.out.rsplit(".", 1)[0] + ".pdf", dpi=400)
    print("->", a.out)
