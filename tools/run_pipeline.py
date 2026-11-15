"""Command-line equivalent of notebooks/filament_pipeline.ipynb: predict every test image with a configuration
and write the submission CSV (same code path: filseg.pipeline.Pipeline).

usage: python tools/run_pipeline.py [--config configs/final_all.json] [--images DIR] [--out submission.csv]
                                    [--max_images N] [--device cuda|cpu]
--images defaults to $FILAMENT_DATA/test/test_images (see src/filseg/paths.py).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from filseg.data import counts_to_mask  # noqa: E402
from filseg.paths import REPO, TEST_IMAGES  # noqa: E402
from filseg.pipeline import Pipeline  # noqa: E402


def check_submission(sub: pd.DataFrame) -> None:
    """Unique ids, every mask non-empty, no overlap between the filaments of one image (the leaderboard rejects it)."""
    assert sub.filament_id.is_unique, "duplicate filament ids"
    for _, g in sub.groupby(sub.filament_id.str.split("_").str[0]):
        acc = np.zeros((2048, 2048), bool)
        for c in g.segmentation_rle:
            m = counts_to_mask(c).astype(bool)
            assert m.any(), "empty mask"
            assert not (acc & m).any(), "overlapping filaments"
            acc |= m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "final_all.json"))
    ap.add_argument("--images", default=TEST_IMAGES)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all (use 2 for a quick smoke test)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = a.config if os.path.isabs(a.config) or os.path.exists(a.config) else os.path.join(REPO, a.config)
    names = sorted(glob.glob(os.path.join(a.images, "*.jpeg")))
    if a.max_images:
        names = names[:a.max_images]
    if not names:
        sys.exit(f"no .jpeg images in {a.images} (set FILAMENT_DATA or --images)")
    pipe = Pipeline(cfg, device=a.device)
    print(f"config {cfg} | device {pipe.device} | {len(names)} images", flush=True)
    rows, t0 = [], time.time()
    for i, path in enumerate(names, 1):
        rows += pipe.rows(path)
        if i % 10 == 0 or i == len(names):
            print(f"{i}/{len(names)} images, {time.time() - t0:.0f} s", flush=True)
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    check_submission(sub)
    sub.to_csv(a.out, index=False)
    print(f"{len(sub)} filaments in {len(names)} images ({len(sub) / len(names):.2f} per image), "
          f"{(time.time() - t0) / len(names):.1f} s per image -> {os.path.abspath(a.out)}; checks OK")
