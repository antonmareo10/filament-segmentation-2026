"""Inter-annotator agreement with the competition metric: on every training image with >= 2 annotators, each
annotator's filaments are scored (exact PQ) against every other annotator of that image, exactly as a prediction
would be. 95% interval: bootstrap over images.

usage: python tools/human_agreement.py [--out human.json]      (reads $FILAMENT_GT_CSV from tools/build_gt.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from filseg.data import counts_to_mask  # noqa: E402
from filseg.metric import GTCache, PQStats  # noqa: E402
from filseg.paths import GT_CSV  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    gt = pd.read_csv(GT_CSV)
    n_ann = gt.groupby("image").annotator_image.nunique()
    gt = gt[gt.image.isin(n_ann[n_ann > 1].index)]
    cache = GTCache(gt.to_dict("records"))
    tot, per_img = PQStats(), []
    for img, g in gt.groupby("image"):
        st = PQStats()
        for ai, ga in g.groupby("annotator_image"):
            lab = np.zeros((2048, 2048), np.int32)  # this annotator's filaments as a "prediction"
            for k, c in enumerate(ga.segmentation_rle, 1):
                m = counts_to_mask(c) > 0
                lab[m & (lab == 0)] = k
            for other, s in cache.score(img, lab, len(ga)).items():
                if other != ai:
                    s.ious = []
                    st.add(s)
        tot.add(st)
        per_img.append((st.tp_iou, st.tp, st.fp, st.fn))
    A = np.array(per_img)
    rng = np.random.RandomState(0)
    boot = []
    for _ in range(1000):
        s = A[rng.randint(0, len(A), len(A))].sum(0)
        boot.append(s[0] / (s[1] + 0.5 * s[2] + 0.5 * s[3]))
    res = {"images": len(A), "PQ": tot.pq, "PQ_ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "SQ": tot.sq, "RQ": tot.rq,
           "TP": tot.tp, "FP": tot.fp, "FN": tot.fn}
    print(f"one annotator vs the others on {len(A)} multi-annotator images: PQ {tot.pq:.4f} "
          f"(95% CI {res['PQ_ci95'][0]:.4f}-{res['PQ_ci95'][1]:.4f}) SQ {tot.sq:.4f} RQ {tot.rq:.4f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
