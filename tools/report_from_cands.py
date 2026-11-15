"""Honest report statistics from out-of-fold candidates (tools/candidates.py extract):
the re-scorer is cross-fitted by fold, so every kept/dropped decision on an image is made by a
model that never saw that image's labels.

usage: python tools/report_from_cands.py --cands c.pkl --folds folds.csv --score_thr 0.325 --out DIR [--rule] [--human_pq 0.340]
Writes DIR/stats.json and DIR/fig_distributions.{png,pdf}: best IoU of every annotated filament, Dice of matched pairs,
one-to-many / many-to-one counts and PQ per GONG site with 95% bootstrap intervals.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filseg.metric import pq_from_iou  # noqa: E402
from rescore_fit import PARAMS, xy  # noqa: E402

SITES = {"Bh": "Big Bear", "Ch": "Cerro Tololo", "Lh": "Learmonth", "Mh": "Mauna Loa", "Th": "El Teide", "Uh": "Udaipur"}


def crossfit_scores(data, folds):
    """Re-scorer output for every candidate, each fold scored by a model fitted on the other folds."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    data = [d for d in data if len(d[1])]
    X, y = xy(data)
    fid = np.concatenate([[folds[img]] * len(f) for img, f, _ in data])
    sc = np.zeros(len(y))
    for k in np.unique(fid):
        sc[fid == k] = HistGradientBoostingRegressor(**PARAMS).fit(X[fid != k], y[fid != k]).predict(X[fid == k])
    out, pos = {}, 0
    for img, f, _ in data:
        out[img] = sc[pos:pos + len(f)]
        pos += len(f)
    return out


def pq(x):
    s = x.sum(0)
    den = s[1] + 0.5 * s[2] + 0.5 * s[3]
    return s[0] / den, s[0] / max(s[1], 1), s[1] / den


def boot_ci(A, rng, n=2000):
    b = np.array([pq(A[rng.randint(0, len(A), len(A))]) for _ in range(n)])
    return np.percentile(b, [2.5, 97.5], axis=0)  # rows: lo, hi; columns: PQ, SQ, RQ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--rule", action="store_true", help="fixed rule area>=500 & peak>=0.9 instead of the re-scorer")
    ap.add_argument("--human_pq", type=float, default=None, help="reference line: one annotator scored against the others")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pickle.load(open(a.cands, "rb"))
    folds = dict(pd.read_csv(a.folds).values)
    scores = None if a.rule else crossfit_scores(d["data"], folds)
    per_img, obs, ious, best_iou, one_many, many_one, n_pred, multi = [], [], [], [], [], [], [], []
    for img, f, mats in d["data"]:
        if len(f) == 0:
            keep = np.zeros(0, bool)
        elif a.rule:
            keep = (np.exp(f[:, 0]) >= 500) & (f[:, 2] >= 0.9)
        else:
            keep = scores[img] >= a.score_thr
        acc = np.zeros(4)
        for m in mats:
            sub = m[:, keep]
            s = pq_from_iou(sub, sub.shape[0], int(keep.sum()))
            acc += [s.tp_iou, s.tp, s.fp, s.fn]
            ious += s.ious
            best_iou += (sub.max(1) if sub.shape[1] else np.zeros(sub.shape[0])).tolist()
            if sub.size:
                one_many += (sub > 0).sum(1).tolist()   # predictions overlapping each annotated filament
                many_one += (sub > 0).sum(0).tolist()   # annotated filaments overlapping each prediction
            else:
                one_many += [0] * sub.shape[0]
                many_one += [0] * sub.shape[1]
        per_img.append(acc)
        obs.append(img[14:16])
        n_pred.append(int(keep.sum()))
        multi.append(len(mats) > 1)
    A, obs, multi = np.array(per_img), np.array(obs), np.array(multi)
    rng = np.random.RandomState(0)
    ci = boot_ci(A, rng)
    ious, best_iou = np.array(ious), np.array(best_iou)
    dices = 2 * ious / (1 + ious)
    P, S, R = pq(A)
    site = {}
    for o in sorted(set(obs)):
        c = boot_ci(A[obs == o], np.random.RandomState(0))
        site[o] = {"name": SITES.get(o, o), "images": int((obs == o).sum()), "PQ": round(float(pq(A[obs == o])[0]), 4),
                   "PQ_ci95": [round(float(c[0, 0]), 4), round(float(c[1, 0]), 4)]}
    stats = {
        "images": len(A), "annotator_images": int(sum(len(m) for _, _, m in d["data"])),
        "PQ": P, "SQ": S, "RQ": R,
        "PQ_ci95": ci[:, 0].tolist(), "SQ_ci95": ci[:, 1].tolist(), "RQ_ci95": ci[:, 2].tolist(),
        "TP_FP_FN": A[:, 1:].sum(0).astype(int).tolist(),
        "pred_per_image": float(np.mean(n_pred)),
        "PQ_multi_annotator_images": float(pq(A[multi])[0]), "n_multi_annotator_images": int(multi.sum()),
        "per_observatory_PQ": {o: v["PQ"] for o, v in site.items()},
        "per_site": site,
        "tp_iou_mean": float(ious.mean()), "tp_dice_mean": float(dices.mean()),
        "tp_iou_median": float(np.median(ious)), "tp_dice_median": float(np.median(dices)),
        "annotated_filaments": int(len(best_iou)),
        "annotated_best_iou_gt_0.5": int((best_iou > 0.5).sum()),
        "annotated_best_iou_0.3_to_0.5": int(((best_iou > 0.3) & (best_iou <= 0.5)).sum()),
        "annotated_not_overlapped": int((best_iou == 0).sum()),
        "gt_overlapped_by_k_preds": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(one_many, 4), return_counts=True))},
        "pred_overlapping_k_gts": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(many_one, 4), return_counts=True))},
        "base_cfg": str(d["cfg"]), "selection": "rule area>=500 & peak>=0.9" if a.rule else f"cross-fitted re-scorer >= {a.score_thr}",
    }
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#b8b6b0", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.8, "axes.labelsize": 6.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 5.8, "axes.edgecolor": INK2, "axes.linewidth": 0.5,
                         "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    # one full-width strip (\textwidth of the two-column template is ~7 in)
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.72), gridspec_kw={"width_ratios": [1, 1, 1.05, 1.15]})
    a0, a1, a2, a3 = ax
    bins = np.linspace(0, 1, 21)
    h, _ = np.histogram(best_iou, bins)
    a0.bar(bins[:-1], h, width=0.05, align="edge", color=[BLUE if b >= 0.5 else GREY for b in bins[:-1]], edgecolor="white", lw=0.4)
    a0.axvline(0.5, color=INK2, lw=0.6, ls="--")
    a0.text(0.53, h.max() * 0.98, f"TP {(best_iou > 0.5).mean():.0%}", color=BLUE, va="top", fontsize=6)
    a0.text(0.47, h.max() * 0.98, f"FN {(best_iou <= 0.5).mean():.0%}", color=INK2, va="top", ha="right", fontsize=6)
    a0.set_xlabel("best IoU with any prediction")
    a0.set_ylabel("annotated filaments")
    a0.set_title("(a) IoU, every annotated filament", loc="left")
    a1.hist(dices, bins=np.linspace(2 / 3, 1, 21), color=BLUE, edgecolor="white", lw=0.4)
    a1.axvline(np.median(dices), color=INK, lw=0.6)
    a1.set_ylim(0, a1.get_ylim()[1] * 1.12)
    a1.text(np.median(dices) + 0.006, a1.get_ylim()[1] * 0.99, f"median {np.median(dices):.2f}", va="top", ha="left", fontsize=6, color=INK)
    a1.set_xlabel("Dice (pairs with IoU > 0.5)")
    a1.set_ylabel("matched pairs")
    a1.set_title("(b) Dice, true positives", loc="left")
    k1 = np.bincount(np.minimum(one_many, 4), minlength=5)
    k2 = np.bincount(np.minimum(many_one, 4), minlength=5)
    x = np.arange(5)
    b1 = a2.bar(x - 0.2, np.maximum(k1, 0.8), 0.38, color=BLUE, label="per annotated filament")
    b2 = a2.bar(x + 0.2, np.maximum(k2, 0.8), 0.38, color=ORANGE, label="per prediction")
    a2.set_yscale("log")
    a2.set_ylim(0.8, 5e6)
    for bars, vals in ((b1, k1), (b2, k2)):
        for bar, v in zip(bars, vals):
            a2.text(bar.get_x() + bar.get_width() / 2, max(v, 0.8) * 1.3, str(v), ha="center", va="bottom", rotation=90, fontsize=5.2, color=INK2)
    a2.set_xticks(x, ["0", "1", "2", "3", "4+"])
    a2.set_xlabel("number of overlapping objects")
    a2.set_ylabel("count (log scale)")
    a2.legend(loc="upper right", frameon=False, handlelength=0.8, borderaxespad=0.0, labelspacing=0.2)
    a2.set_title("(c) one-to-many / many-to-one", loc="left")
    names = list(site)
    y = np.arange(len(names))
    vals = np.array([site[o]["PQ"] for o in names])
    lo = np.array([site[o]["PQ_ci95"][0] for o in names])
    hi = np.array([site[o]["PQ_ci95"][1] for o in names])
    a3.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="o", ms=3, color=BLUE, ecolor=BLUE, elinewidth=0.8, capsize=1.5)
    a3.axvline(P, color=INK, lw=0.6)
    a3.text(P - 0.002, len(names) - 0.45, f"all {P:.3f}", ha="right", fontsize=5.8, color=INK)
    xmin = 0.39
    if a.human_pq:
        xmin = a.human_pq - 0.012
        a3.axvline(a.human_pq, color=ORANGE, lw=0.6, ls="--")
        a3.text(a.human_pq + 0.003, -0.6, f"expert vs\nexpert {a.human_pq:.2f}", fontsize=5.8, color=ORANGE, va="bottom")
    a3.set_xlim(xmin, max(hi) + 0.01)
    a3.set_yticks(y, [f"{site[o]['name']} {site[o]['images']}" for o in names])
    a3.set_ylim(-0.7, len(names) + 0.1)
    a3.set_xlabel("PQ, 95% bootstrap CI")
    a3.set_title("(d) PQ per site (images)", loc="left")
    for z in ax.ravel():
        z.tick_params(length=2, pad=1.5)
    plt.tight_layout(pad=0.2, w_pad=0.9)
    plt.savefig(os.path.join(a.out, "fig_distributions.png"), dpi=300)
    plt.savefig(os.path.join(a.out, "fig_distributions.pdf"))
    print("->", a.out)
