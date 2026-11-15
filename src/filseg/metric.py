"""Exact re-implementation of the organisers' Panoptic Quality (self-evaluation notebook).

Official procedure (azimahmadzadeh/self-evaluation-notebook):
  * iterate over every GT *annotator-image* (an image annotated by k annotators counts k times);
  * predictions of that image (filament_id prefix before the first '_') are compared with
    that annotator's GT filaments;
  * IoU matrix from binary masks; a pair is TP when IoU > 0.5 (strict);
  * FP = predictions with no TP pair, FN = GT filaments with no TP pair;
  * PQ = sum(IoU of TP pairs) / (TP + 0.5 FP + 0.5 FN), pooled over all annotator-images.

The leaderboard rejects overlapping predictions, so predictions can be represented as one
integer label map per image; intersections are then exact integer counts (bincount).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import counts_to_mask

IOU_THR = 0.5


@dataclass
class PQStats:
    tp_iou: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ious: list = field(default_factory=list)

    def add(self, other: "PQStats") -> None:
        self.tp_iou += other.tp_iou
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.ious.extend(other.ious)

    @property
    def pq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp_iou / d if d > 0 else 0.0

    @property
    def sq(self) -> float:
        return self.tp_iou / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d > 0 else 0.0


def iou_matrix_from_labels(gt_masks: list[np.ndarray], pred_labels: np.ndarray, n_pred: int) -> np.ndarray:
    """gt_masks: list of HxW {0,1}; pred_labels: HxW int (0 = background, 1..n_pred)."""
    pred_areas = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)[1:].astype(np.int64)
    m = np.zeros((len(gt_masks), n_pred), dtype=np.float64)
    for i, g in enumerate(gt_masks):
        g = g.astype(bool)
        inter = np.bincount(pred_labels[g], minlength=n_pred + 1)[1:].astype(np.int64)
        union = int(g.sum()) + pred_areas - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            m[i] = np.where(union == 0, 0.0, inter / np.maximum(union, 1))
    return m


def pq_from_iou(iou: np.ndarray, n_gt: int, n_pred: int) -> PQStats:
    s = PQStats()
    if n_gt == 0:
        s.fp = n_pred
        return s
    if n_pred == 0:
        s.fn = n_gt
        return s
    hit = iou > IOU_THR
    vals = iou[hit]
    s.tp = int(hit.sum())
    s.tp_iou = float(vals.sum())
    s.ious = vals.tolist()
    s.fp = int((hit.sum(axis=0) == 0).sum())
    s.fn = int((hit.sum(axis=1) == 0).sum())
    return s


def labels_from_counts(pred_counts: list[str]) -> tuple[np.ndarray, int]:
    lab = None
    for k, c in enumerate(pred_counts, start=1):
        m = counts_to_mask(c)
        if lab is None:
            lab = np.zeros(m.shape, dtype=np.int32)
        if (lab[m > 0] != 0).any():
            raise ValueError("overlapping predictions are rejected by the leaderboard")
        lab[m > 0] = k
    if lab is None:
        lab = np.zeros((2048, 2048), dtype=np.int32)
    return lab, len(pred_counts)


def score_image(gt_masks_by_annotator: dict[str, list[np.ndarray]], pred_labels: np.ndarray, n_pred: int) -> dict[str, PQStats]:
    """Score one image's prediction against each annotator's GT of that image."""
    out = {}
    for ann_img, gts in gt_masks_by_annotator.items():
        iou = iou_matrix_from_labels(gts, pred_labels, n_pred) if gts else np.zeros((0, n_pred))
        out[ann_img] = pq_from_iou(iou, len(gts), n_pred)
    return out


class GTCache:
    """Per-image GT masks stored as bbox crops for fast exact intersection counting."""

    def __init__(self, gt_rows):
        from pycocotools import mask as mask_util
        from collections import defaultdict
        self.by_image = defaultdict(dict)
        for r in gt_rows:
            rle = {"size": [2048, 2048], "counts": r["segmentation_rle"].encode("ascii")}
            x, y, w, h = [int(v) for v in mask_util.toBbox(rle)]
            m = mask_util.decode(rle)
            y1, x1 = min(2048, y + h + 1), min(2048, x + w + 1)
            crop = m[y:y1, x:x1].astype(bool)
            self.by_image[r["image"]].setdefault(r["annotator_image"], []).append((y, y1, x, x1, crop, int(crop.sum())))

    def iou_matrices(self, image: str, pred_labels: np.ndarray, n_pred: int) -> dict[str, np.ndarray]:
        pred_areas = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)[1:].astype(np.int64)
        out = {}
        for ai, gts in self.by_image[image].items():
            iou = np.zeros((len(gts), n_pred))
            for i, (y0, y1, x0, x1, crop, area) in enumerate(gts):
                inter = np.bincount(pred_labels[y0:y1, x0:x1][crop], minlength=n_pred + 1)[1:].astype(np.int64)
                union = area + pred_areas - inter
                iou[i] = np.where(union == 0, 0.0, inter / np.maximum(union, 1))
            out[ai] = iou
        return out

    def score(self, image: str, pred_labels: np.ndarray, n_pred: int) -> dict[str, PQStats]:
        return {ai: pq_from_iou(m, m.shape[0], n_pred)
                for ai, m in self.iou_matrices(image, pred_labels, n_pred).items()}


def pq_subset(iou_mats: list[np.ndarray], keeps: list[np.ndarray]) -> PQStats:
    """Exact PQ when only a subset of candidates is submitted. Dropping a candidate never changes
    the IoU of the others (instances are disjoint), so each image's IoU matrix is simply column-sliced."""
    tot = PQStats()
    for m, k in zip(iou_mats, keeps):
        sub = m[:, k]
        s = pq_from_iou(sub, sub.shape[0], int(k.sum()))
        s.ious = []
        tot.add(s)
    return tot
