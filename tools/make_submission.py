"""Build the Kaggle submission CSV from TEST probability maps.

usage: python tools/make_submission.py --probs DIR[,DIR2...] --cfg "thr=0.5;seed_thr=0.6;min_area=300" --out sub.csv
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from filseg.data import mask_to_counts  # noqa: E402
from filseg.postproc import instances, upsample  # noqa: E402
from filseg.rescore import component_features  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_probs import load_prob, parse_grid  # noqa: E402

from filseg.paths import TEST_IMAGES as TEST_DIR  # noqa: E402  ($FILAMENT_DATA/test/test_images)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rescorer", default="", help="pickle from tools/rescore_fit.py; --cfg then gives the base binarisation")
    ap.add_argument("--score_thr", type=float, default=0.3)
    a = ap.parse_args()
    resc = pickle.load(open(a.rescorer, "rb")) if a.rescorer else None
    dirs = a.probs.split(",")
    cfg = parse_grid(a.cfg)
    assert len(cfg) == 1
    cfg = cfg[0]
    names = sorted(f.rsplit(".", 1)[0] for f in os.listdir(TEST_DIR))
    rows, empty = [], 0
    for img in names:
        prob = upsample(load_prob(dirs, img))
        lab, n = instances(prob, cfg)
        if resc is not None and n:
            keep = resc["model"].predict(component_features(prob, lab, n)) >= a.score_thr
            remap = np.zeros(n + 1, np.int32)
            remap[1:][keep] = np.arange(1, keep.sum() + 1)
            lab, n = remap[lab], int(keep.sum())
        if n == 0:
            empty += 1
        for k in range(1, n + 1):
            rows.append({"filament_id": f"{img}_{k}", "segmentation_rle": mask_to_counts(lab == k)})
    df = pd.DataFrame(rows)
    assert df.filament_id.is_unique and set(df.filament_id.str.split("_").str[0]) <= set(names)
    df.to_csv(a.out, index=False)
    print(f"{cfg} -> {len(df)} filaments in {len(names)} images ({empty} empty), mean {len(df)/len(names):.2f}/img -> {a.out}")
