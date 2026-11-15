"""Ground-truth loading and RLE helpers (COCO-style annotations of MAGFiLO)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

H = W = 2048


def load_coco(json_path: str | Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def polygon_to_rle(polygon: list[float], h: int = H, w: int = W) -> dict:
    """Polygon -> compressed RLE exactly as pycocotools' COCO.annToRLE does."""
    rles = mask_util.frPyObjects([polygon], h, w)
    return mask_util.merge(rles)


def rle_counts(rle: dict) -> str:
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def counts_to_mask(counts: str, h: int = H, w: int = W) -> np.ndarray:
    return mask_util.decode({"size": [h, w], "counts": counts.encode("ascii")})


def mask_to_counts(mask: np.ndarray) -> str:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle_counts(rle)


def build_gt_table(coco: dict) -> list[dict]:
    """One row per GT filament, in the submission-like format used by the official
    self-evaluation notebook: filament_id = '<annotator>-<image>_<k>'."""
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    rows = []
    for im in coco["images"]:
        for k, a in enumerate(anns_by_img.get(im["id"], []), start=1):
            rle = polygon_to_rle(a["segmentation"][0], im["height"], im["width"])
            rows.append({
                "filament_id": f"{im['id']}_{k}",
                "annotator_image": im["id"],
                "image": im["file_name"].rsplit(".", 1)[0],
                "segmentation_rle": rle_counts(rle),
                "category_id": a["category_id"],
            })
    return rows


def image_name(annotator_image: str) -> str:
    return annotator_image.split("-", 1)[1]
