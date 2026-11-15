"""Write the ground-truth table used by the evaluation tools (one row per annotated filament,
filament_id = '<annotator>-<image>_<k>', COCO RLE counts exactly as pycocotools' annToRLE).

usage: python tools/build_gt.py [--out gt_train.csv]   (reads $FILAMENT_DATA/train/...json)
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filseg.data import build_gt_table, load_coco  # noqa: E402
from eval_probs import DATA, GT_CSV  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=GT_CSV)
    a = ap.parse_args()
    coco = load_coco(os.path.join(DATA, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    df = pd.DataFrame(build_gt_table(coco))
    df.to_csv(a.out, index=False)
    print(f"{len(df)} filaments, {df.annotator_image.nunique()} annotator-images, {df.image.nunique()} images -> {a.out}")
