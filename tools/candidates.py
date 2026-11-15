"""Extract candidate instances (+features, +exact IoU vs every annotator) from OOF probability maps,
then tune instance filtering exactly and fast (dropping candidates never changes other IoUs).

usage:
  python tools/candidates.py extract --probs DIR[,DIR2] --thr 0.4 [--close_r 0] --out cands.pkl
  python tools/candidates.py tune --cands cands.pkl --folds folds.csv
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filseg.metric import GTCache, pq_subset  # noqa: E402
from filseg.postproc import PostCfg, instances, upsample  # noqa: E402
from filseg.rescore import FEATURES, component_features  # noqa: E402
from eval_probs import GT_CSV, load_prob  # noqa: E402

_G = {}


def _init(images):
    gt = pd.read_csv(GT_CSV)
    _G["cache"] = GTCache(gt[gt.image.isin(images)].to_dict("records"))


def _extract(args):
    dirs, img, cfg = args
    p = upsample(load_prob(dirs, img))
    lab, n = instances(p, cfg)
    feats = component_features(p, lab, n)
    mats = _G["cache"].iou_matrices(img, lab, n)
    return img, feats, [mats[a] for a in sorted(mats)]


def extract(dirs, cfg, images, workers=4):
    with Pool(workers, initializer=_init, initargs=(images,)) as pool:
        return pool.map(_extract, [(dirs, im, cfg) for im in images], chunksize=4)


def rule_grid(data):
    """Exact PQ for simple keep rules: area >= a and p_max >= s and p_mean >= m."""
    res = []
    for a in [0, 200, 400, 600, 800, 1000, 1500, 2000]:
        for s in [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97]:
            for m in [0.0, 0.5, 0.6, 0.7]:
                mats, keeps = [], []
                for img, f, ms in data:
                    k = (np.exp(f[:, 0]) >= a) & (f[:, 2] >= s) & (f[:, 1] >= m) if len(f) else np.zeros(0, bool)
                    for mm in ms:
                        mats.append(mm)
                        keeps.append(k)
                st = pq_subset(mats, keeps)
                res.append((a, s, m, st.pq, st.sq, st.rq, st.tp, st.fp, st.fn))
    return pd.DataFrame(res, columns=["min_area", "seed_thr", "min_pmean", "PQ", "SQ", "RQ", "TP", "FP", "FN"])


def learned(data, folds: dict, seed=0):
    """Cross-fitted gradient boosting: P(candidate is TP for an annotator) and the PQ vs score threshold."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, y, fid = [], [], []
    for img, f, ms in data:
        if len(f) == 0:
            continue
        hits = np.zeros(len(f))
        for mm in ms:
            if mm.shape[0]:
                hits += (mm.max(0) > 0.5)
        X.append(f)
        y.append(hits / len(ms))
        fid += [folds[img]] * len(f)
    X, y, fid = np.concatenate(X), np.concatenate(y), np.array(fid)
    score = np.zeros(len(y))
    for k in np.unique(fid):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=40, random_state=seed)
        m.fit(X[fid != k], y[fid != k])
        score[fid == k] = m.predict(X[fid == k])
    res, pos = [], 0
    per_img = []
    for img, f, ms in data:
        per_img.append(score[pos:pos + len(f)])
        pos += len(f)
    for t in np.arange(0.05, 0.75, 0.025):
        mats, keeps = [], []
        for (img, f, ms), sc in zip(data, per_img):
            k = sc >= t
            for mm in ms:
                mats.append(mm)
                keeps.append(k)
        st = pq_subset(mats, keeps)
        res.append((round(t, 3), st.pq, st.sq, st.rq, st.tp, st.fp, st.fn))
    return pd.DataFrame(res, columns=["score_thr", "PQ", "SQ", "RQ", "TP", "FP", "FN"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "tune"])
    ap.add_argument("--probs")
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--close_r", type=int, default=0)
    ap.add_argument("--dilate_r", type=int, default=0)
    ap.add_argument("--group_r", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--cands")
    ap.add_argument("--folds")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.cmd == "extract":
        dirs = a.probs.split(",")
        imgs = sorted(set.intersection(*[{f[:-4] for f in os.listdir(d) if f.endswith(".png")} for d in dirs]))
        cfg = PostCfg(thr=a.thr, seed_thr=0.0, min_area=0, close_r=a.close_r, dilate_r=a.dilate_r, group_r=a.group_r)
        data = extract(dirs, cfg, imgs, a.workers)
        pickle.dump({"cfg": cfg, "data": data, "features": FEATURES}, open(a.out, "wb"))
        print(f"{len(imgs)} images, {sum(len(f) for _, f, _ in data)} candidates -> {a.out}")
    else:
        d = pickle.load(open(a.cands, "rb"))
        print("base cfg", d["cfg"])
        g = rule_grid(d["data"]).sort_values("PQ", ascending=False)
        print(g.head(8).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if a.folds:
            folds = dict(pd.read_csv(a.folds).values)
            print(learned(d["data"], folds).sort_values("PQ", ascending=False).head(6).to_string(
                index=False, float_format=lambda x: f"{x:.4f}"))
