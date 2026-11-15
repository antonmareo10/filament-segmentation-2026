"""Default data locations for the command-line tools, overridable with environment variables.

FILAMENT_DATA    the unzipped competition folder ``MAGFiLO_1.0_Kaggle_2026`` (with train/ and test/)
                 default: <repo>/data/MAGFiLO_1.0_Kaggle_2026
FILAMENT_GT_CSV  ground-truth table of the training images written by ``tools/build_gt.py``
                 default: <repo>/data/gt_train.csv
FILAMENT_RUNS    outputs of the training kernels (out-of-fold / test probability maps, candidate files)
                 default: <repo>/runs

The notebook does not need any of them on Kaggle (it finds the competition data under /kaggle/input).
"""
from __future__ import annotations

import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.environ.get("FILAMENT_DATA", os.path.join(REPO, "data", "MAGFiLO_1.0_Kaggle_2026"))
GT_CSV = os.environ.get("FILAMENT_GT_CSV", os.path.join(REPO, "data", "gt_train.csv"))
RUNS = os.environ.get("FILAMENT_RUNS", os.path.join(REPO, "runs"))
TRAIN_JSON = os.path.join(DATA, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
TRAIN_IMAGES = os.path.join(DATA, "train", "train_images")
TEST_IMAGES = os.path.join(DATA, "test", "test_images")
