"""Predict filament instances for a folder of H-alpha JPEGs with saved model weights.

usage: python tools/predict.py --images DIR --weights w1.pth[,w2.pth...] --size 1024 \
           --cfg "thr=0.5;seed_thr=0.9;min_area=500" --out submission.csv [--probs-out DIR] [--device cuda]

Writes the Kaggle submission format (filament_id, segmentation_rle). Works on CPU (slow) or GPU.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filseg.data import mask_to_counts  # noqa: E402
from filseg.model import build_model, load_image, load_weights, predict_tta  # noqa: E402
from filseg.postproc import instances, upsample  # noqa: E402
from eval_probs import parse_grid  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probs-out", default="")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = parse_grid(a.cfg)[0]
    models = [load_weights(build_model(a.encoder, a.arch), w, a.device) for w in a.weights.split(",")]
    names = sorted(f.rsplit(".", 1)[0] for f in os.listdir(a.images) if f.lower().endswith((".jpeg", ".jpg")))
    rows = []
    for i, n in enumerate(names):
        prob = predict_tta(models, load_image(os.path.join(a.images, n + ".jpeg"), a.size), a.device)
        if a.probs_out:
            os.makedirs(a.probs_out, exist_ok=True)
            cv2.imwrite(os.path.join(a.probs_out, n + ".png"), (prob * 255 + 0.5).clip(0, 255).astype("uint8"))
        lab, k = instances(upsample((prob * 255 + 0.5).clip(0, 255).astype("uint8")), cfg)
        rows += [{"filament_id": f"{n}_{j}", "segmentation_rle": mask_to_counts(lab == j)} for j in range(1, k + 1)]
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(names)} images", flush=True)
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(a.out, index=False)
    print(f"{len(rows)} filaments in {len(names)} images -> {a.out}")
