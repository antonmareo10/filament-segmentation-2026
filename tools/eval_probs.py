"""Score probability maps of TRAIN images (out-of-fold) with the exact PQ, for a grid of
post-processing settings.

usage: python tools/eval_probs.py --probs DIR[,DIR2...] [--grid "thr=0.4,0.5;min_area=0,200"] [--workers 4]
Several prob dirs are averaged (ensembling). Only images present in every dir are scored.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from filseg.metric import GTCache, PQStats  # noqa: E402
from filseg.postproc import PostCfg, instances, upsample  # noqa: E402
from filseg.paths import DATA, GT_CSV  # noqa: E402,F401  (env FILAMENT_DATA / FILAMENT_GT_CSV, see filseg/paths.py)
_G = {}


def _init(gt_csv, images):
    gt = pd.read_csv(gt_csv)
    gt = gt[gt.image.isin(images)]
    _G["cache"] = GTCache(gt.to_dict("records"))


def load_prob(dirs, img, weights=None):
    """Average of the probability PNGs of several runs (resized to 2048 when resolutions differ)."""
    ps = [cv2.imread(os.path.join(d, img + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for d in dirs]
    if len({p.shape for p in ps}) > 1:
        ps = [upsample(p) for p in ps]
    w = weights or [1.0] * len(ps)
    return sum(wi * p for wi, p in zip(w, ps)) / sum(w)


def _work(args):
    dirs, img, cfgs = args
    p = upsample(load_prob(dirs, img))
    res = []
    for c in cfgs:
        lab, n = instances(p, c)
        st = PQStats()
        for s in _G["cache"].score(img, lab, n).values():
            st.ious = []
            st.add(s)
        res.append((st.tp_iou, st.tp, st.fp, st.fn))
    return img, res


def parse_grid(g: str):
    keys, vals = [], []
    for part in filter(None, g.split(";")):
        k, v = part.split("=")
        typ = type(getattr(PostCfg(), k))
        keys.append(k)
        vals.append([typ(float(x)) if typ is not bool else x in ("1", "True", "true") for x in v.split(",")])
    return [PostCfg(**dict(zip(keys, combo))) for combo in itertools.product(*vals)] or [PostCfg()]


def evaluate(dirs, cfgs, images, workers=4, gt_csv=GT_CSV):
    with Pool(workers, initializer=_init, initargs=(gt_csv, images)) as pool:
        out = pool.map(_work, [(dirs, im, cfgs) for im in images], chunksize=4)
    return out


def summarise(out, cfgs, by=None):
    rows = []
    for j, c in enumerate(cfgs):
        groups = {"ALL": out} if by is None else {}
        if by is not None:
            for img, r in out:
                groups.setdefault(by(img), []).append((img, r))
        for gname, items in groups.items():
            a = np.array([r[j] for _, r in items], dtype=np.float64).sum(0)
            d = a[1] + 0.5 * a[2] + 0.5 * a[3]
            rows.append({**c.as_dict(), "group": gname, "PQ": a[0] / d, "SQ": a[0] / max(a[1], 1), "RQ": a[1] / d,
                         "TP": int(a[1]), "FP": int(a[2]), "FN": int(a[3])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--grid", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--by-obs", action="store_true")
    ap.add_argument("--images", default="", help="optional csv with column image (subset)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dirs = a.probs.split(",")
    imgs = sorted(set.intersection(*[{f[:-4] for f in os.listdir(d) if f.endswith(".png")} for d in dirs]))
    if a.images:
        imgs = sorted(set(imgs) & set(pd.read_csv(a.images).image))
    cfgs = parse_grid(a.grid)
    print(f"{len(imgs)} images, {len(cfgs)} configs", flush=True)
    out = evaluate(dirs, cfgs, imgs, a.workers)
    df = summarise(out, cfgs)
    pd.set_option("display.width", 200)
    print(df.sort_values("PQ", ascending=False).head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if a.by_obs:
        best = cfgs[int(df.PQ.values.argmax())]
        print("by observatory for best cfg:")
        print(summarise(out, [best], by=lambda im: im[14:16]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if a.out:
        df.to_csv(a.out, index=False)
