"""Fit the instance re-scorer on out-of-fold candidates and save it.

usage: python tools/rescore_fit.py --cands oof_cands.pkl --out rescorer.pkl
The target of each candidate is the fraction of the image's annotators for which it is a true
positive (IoU > 0.5). The score threshold is chosen beforehand with cross-fitting
(`tools/candidates.py tune --folds ...`); here the final model is refit on all OOF candidates.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from filseg.rescore import FEATURES  # noqa: E402,F401  (needed to unpickle candidates)

PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)


def xy(data):
    X, y = [], []
    for _, f, ms in data:
        if len(f) == 0:
            continue
        hits = np.zeros(len(f))
        for m in ms:
            if m.shape[0]:
                hits += m.max(0) > 0.5
        X.append(f)
        y.append(hits / len(ms))
    return np.concatenate(X), np.concatenate(y)


if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingRegressor
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pickle.load(open(a.cands, "rb"))
    X, y = xy(d["data"])
    m = HistGradientBoostingRegressor(**PARAMS).fit(X, y)
    pickle.dump({"model": m, "base_cfg": d["cfg"], "features": d["features"]}, open(a.out, "wb"))
    print(f"fit on {len(y)} candidates (mean target {y.mean():.3f}) base {d['cfg']} -> {a.out}")
