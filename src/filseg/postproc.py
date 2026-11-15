"""Probability map -> non-overlapping filament instances.

Pipeline (all parameters tuned on out-of-fold predictions with the exact PQ metric):
  1. bilinear upsampling of the model probability to the native 2048 x 2048 grid;
  2. binarisation at ``thr`` (low threshold of a hysteresis pair);
  3. optional morphological closing (radius ``close_r``) to reconnect fragments of one filament;
  4. connected components (8-connectivity) -> candidate instances; with ``group_r`` > 0, components
     closer than ~2*group_r px are grouped into one instance without adding pixels;
  5. a component is kept only if its area >= ``min_area`` and its peak probability >= ``seed_thr``;
  6. holes are filled (MAGFiLO polygons are simple, hole-free shapes).
The result is an int32 label map (0 = background) so instances never overlap.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage

NATIVE = 2048


@dataclass(frozen=True)
class PostCfg:
    thr: float = 0.5
    seed_thr: float = 0.5
    min_area: int = 0
    close_r: int = 0
    fill_holes: bool = True
    dilate_r: int = 0
    group_r: int = 0  # merge components closer than ~2*group_r px into one instance (pixels unchanged)

    def as_dict(self):
        return asdict(self)


def upsample(prob: np.ndarray, size: int = NATIVE) -> np.ndarray:
    """prob: float32 or uint8 (0..255) map at model resolution -> float32 [0,1] at native size."""
    p = prob.astype(np.float32)
    if prob.dtype == np.uint8:
        p /= 255.0
    if p.shape[0] != size:
        p = cv2.resize(p, (size, size), interpolation=cv2.INTER_LINEAR)
    return p


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes of a binary uint8 mask (holes = background regions not connected to the border)."""
    h, w = mask.shape
    ff = mask.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    return mask | (1 - ff)


def instances(prob_native: np.ndarray, cfg: PostCfg) -> tuple[np.ndarray, int]:
    binm = (prob_native >= cfg.thr).astype(np.uint8)
    if cfg.close_r > 0:
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, _disk(cfg.close_r))
    if cfg.dilate_r > 0:
        binm = cv2.dilate(binm, _disk(cfg.dilate_r))
    if cfg.fill_holes:
        binm = _fill_holes(binm)
    if cfg.group_r > 0:
        grown = cv2.dilate(binm, _disk(cfg.group_r))
        n, glab = cv2.connectedComponents(grown, connectivity=8)
        lab = np.where(binm > 0, glab, 0).astype(np.int32)
        areas = np.bincount(lab.ravel(), minlength=n)[1:]
    else:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.zeros(0, np.int64)
    if n <= 1:
        return np.zeros(binm.shape, np.int32), 0
    peak = np.asarray(ndimage.maximum(prob_native, lab, index=np.arange(1, n)), np.float32)
    keep = (areas >= cfg.min_area) & (peak >= cfg.seed_thr)
    remap = np.zeros(n, np.int32)
    remap[1:][keep] = np.arange(1, keep.sum() + 1, dtype=np.int32)
    return remap[lab], int(keep.sum())
