"""Instance-level re-scoring of candidate filaments.

PQ charges the same 0.5 in the denominator for every extra prediction, whether it turns into a
TP or an FP, so a candidate is worth keeping when P(TP) * E[IoU | TP] > PQ / 2.  A pixel-wise
network does not output P(TP) per instance, so we learn it from out-of-fold candidates:

  features of each connected component (size, shape, probability statistics, position on disk)
  -> target = fraction of the image's annotators for which the component is a TP (IoU > 0.5)

and keep candidates whose predicted score exceeds a threshold tuned on the OOF PQ.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

FEATURES = [
    "log_area", "p_mean", "p_max", "p_q90", "bbox_fill", "elong", "log_perim_ratio",
    "r_disk", "n_cand", "frac_hi", "p_sum_log",
]


def component_features(prob: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    """prob: native-resolution probability (float32), lab: int label map (1..n)."""
    if n == 0:
        return np.zeros((0, len(FEATURES)), np.float32)
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    psum = np.asarray(ndimage.sum(prob, lab, idx))
    pmax = np.asarray(ndimage.maximum(prob, lab, idx))
    hi = np.asarray(ndimage.sum(prob >= 0.7, lab, idx))
    objs = ndimage.find_objects(lab)
    h, w = lab.shape
    rows = []
    for k, sl in enumerate(objs, start=1):
        sub = lab[sl] == k
        pv = prob[sl][sub]
        ys, xs = np.nonzero(sub)
        bh, bw = sub.shape
        if len(ys) > 2:
            cov = np.cov(np.stack([ys, xs]).astype(np.float64))
            ev = np.sort(np.linalg.eigvalsh(cov))
            elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
        else:
            elong = 1.0
        # boundary length proxy: pixels with a background 4-neighbour
        er = ndimage.binary_erosion(sub)
        perim = float(sub.sum() - er.sum())
        cy = (sl[0].start + ys.mean()) / h - 0.5
        cx = (sl[1].start + xs.mean()) / w - 0.5
        a = area[k - 1]
        rows.append([
            np.log(a), psum[k - 1] / a, pmax[k - 1], float(np.quantile(pv, 0.9)), a / (bh * bw), elong,
            np.log((perim + 1) / np.sqrt(a)), np.sqrt(cx * cx + cy * cy) / 0.45, n, hi[k - 1] / a,
            np.log1p(psum[k - 1]),
        ])
    return np.asarray(rows, np.float32)


def component_targets(iou_by_annotator: list[np.ndarray]) -> np.ndarray:
    """iou_by_annotator: list over annotators of (n_gt, n_pred) IoU matrices of one image.
    Returns per-candidate fraction of annotators for which it is a TP, and its mean TP IoU."""
    n_pred = iou_by_annotator[0].shape[1] if iou_by_annotator else 0
    tp = np.zeros(n_pred)
    iou_sum = np.zeros(n_pred)
    for m in iou_by_annotator:
        if m.shape[0] == 0:
            continue
        best = m.max(axis=0)
        hit = best > 0.5
        tp += hit
        iou_sum += np.where(hit, best, 0)
    k = max(len(iou_by_annotator), 1)
    return tp / k, np.where(tp > 0, iou_sum / np.maximum(tp, 1), 0)
