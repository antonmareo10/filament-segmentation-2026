"""Pipeline diagram for the report: training lane (top) and inference lane (bottom) of the final model
(configs/final_all.json). Writes report/fig_pipeline.{pdf,png}.

usage: python tools/fig_pipeline.py [--out report/fig_pipeline]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2 = "#0b0b0b", "#52514e"
TRAIN_FC, TRAIN_EC = "#e3edf9", "#2a78d6"   # training lane (blue)
INFER_FC, INFER_EC = "#fbe6dc", "#eb6834"   # inference lane (orange)
OUT_FC = "#efeeea"

TRAIN = [
    ("Training data", "707 H$\\alpha$ images\n1–3 annotators each\n8,199 filaments"),
    ("Soft target", "mean of the\nannotators' masks\n= expert consensus"),
    ("Day-grouped CV", "5 folds, a day is\nnever split; every\nscore is out-of-fold"),
    ("2 U-Nets × 5 folds", "ResNet-34 and\nEffNetV2-S @1536$^2$\nBCE+Dice, 25 epochs"),
    ("OOF candidates", "components of OOF\nmaps: 11 features,\nshare of experts hit"),
    ("Instance re-scorer", "gradient boosting,\ncross-fitted by fold,\nthreshold by OOF PQ"),
]
INFER = [
    ("Test image", "H$\\alpha$ JPEG 2048$^2$\ngrayscale", OUT_FC, INK2),
    ("Normalise", "resize 1536 / 2048\n(disk − median) / IQR\noff-disk constant", INFER_FC, INFER_EC),
    ("Ensemble + TTA", "10 U-Nets × 12 views\n8 dihedral @1536\n+ 4 flips @2048", INFER_FC, INFER_EC),
    ("Instances", "mean prob. @2048$^2$\n$p>0.5$, fill holes,\njoin gaps < 20 px", INFER_FC, INFER_EC),
    ("Re-score", "11 features per\ncandidate; keep if\nscore $\\geq 0.325$", INFER_FC, INFER_EC),
    ("Submission", "disjoint masks\n→ COCO RLE\n6.3 per test image", OUT_FC, INK2),
]

def box(ax, x, y, w, h, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=6.3, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.27, body, ha="center", va="top", fontsize=5.75, color=INK, linespacing=1.15)


def arrow(ax, p, q, color=INK2, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("report", "fig_pipeline"))
    a = ap.parse_args()
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    n, w, gap, h = 6, 1.0, 0.17, 0.82
    W = n * w + (n - 1) * gap
    fig = plt.figure(figsize=(7.0, 2.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.33, W + 0.02)
    ax.set_ylim(-0.02, 2.12)
    ax.axis("off")
    yt, yi = 1.22, 0.06
    ax.text(-0.3, yt + h / 2, "TRAINING", rotation=90, ha="center", va="center", fontsize=6.4, color=TRAIN_EC, fontweight="bold")
    ax.text(-0.3, yi + h / 2, "INFERENCE", rotation=90, ha="center", va="center", fontsize=6.4, color=INFER_EC, fontweight="bold")
    for i, (t, b) in enumerate(TRAIN):
        x = i * (w + gap)
        box(ax, x, yt, w, h, t, b, TRAIN_FC, TRAIN_EC)
        if i < n - 1:
            arrow(ax, (x + w + 0.01, yt + h / 2), (x + w + gap - 0.01, yt + h / 2))
    for i, (t, b, fc, ec) in enumerate(INFER):
        x = i * (w + gap)
        box(ax, x, yi, w, h, t, b, fc, ec)
        if i < n - 1:
            arrow(ax, (x + w + 0.01, yi + h / 2), (x + w + gap - 0.01, yi + h / 2))
    # trained weights -> ensemble; fitted re-scorer -> re-score step
    xu = 3 * (w + gap) + w / 2
    xe = 2 * (w + gap) + w / 2
    arrow(ax, (xu - 0.25, yt - 0.01), (xe + 0.2, yi + h + 0.01), color=TRAIN_EC, ls="--")
    ax.text((xu + xe) / 2 + 0.06, (yt + yi + h) / 2, "10 weight files", fontsize=5.6, color=TRAIN_EC, ha="left", va="center")
    xr = 5 * (w + gap) + w / 2
    xs = 4 * (w + gap) + w / 2
    arrow(ax, (xr - 0.25, yt - 0.01), (xs + 0.2, yi + h + 0.01), color=TRAIN_EC, ls="--")
    ax.text((xr + xs) / 2 + 0.06, (yt + yi + h) / 2, "fitted model", fontsize=5.6, color=TRAIN_EC, ha="left", va="center")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(a.out + ".pdf", bbox_inches="tight", pad_inches=0.01)
    plt.savefig(a.out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("->", a.out + ".pdf/.png")
