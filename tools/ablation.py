"""Ablation table with uncertainty, computed from out-of-fold candidate files (tools/candidates.py extract).

Every row is scored the same way, on out-of-fold predictions only:
  * exact PQ / SQ / RQ pooled over all annotator-images of the images in the candidate file;
  * instances are selected by the cross-fitted re-scorer (each fold's candidates are scored by a GBM fitted on the
    other four folds) at a fixed threshold, at the best threshold of a fixed grid ("best"), by the fixed rule
    area >= 500 px and peak probability >= 0.9 ("rule"), or all candidates are kept ("all");
  * "best" rows also report a nested estimate: the threshold of each fold is chosen on the other four folds;
  * 95% intervals bootstrap the images (2,000 resamples); a row with a "ref" also reports its difference to that
    row with a *paired* bootstrap on the common images (the same resampled images for both rows);
  * PQ' is the expected score when a test annotator draws only a fraction r of the filaments a training annotator
    would draw (missed filaments turn our true positives into false positives):
        PQ'(r) = r*S / (r*TP + FP/2 + (1-r)*TP/2 + r*FN/2),   S = sum of the IoUs of the true positives.
    For rows with a public leaderboard score (shown with two decimals) the script reports the interval of r that
    reproduces the displayed value, and the intersection over all submissions.

usage: python tools/ablation.py --spec report/ablation_spec.json --runs DIR --folds configs/folds.csv --out report/ablation.json
       (--runs: folder holding the candidate files named in the spec; default $FILAMENT_RUNS)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filseg.metric import pq_from_iou  # noqa: E402
from report_from_cands import crossfit_scores  # noqa: E402

GRID = np.round(np.arange(0.2, 0.601, 0.025), 3)
B = 2000


def pq_of(acc: np.ndarray) -> tuple[float, float, float]:
    """acc: (n_images, >=4) = sum IoU of TPs, TP, FP, FN, ... -> pooled PQ, SQ, RQ."""
    s = acc.sum(0)
    den = s[1] + 0.5 * s[2] + 0.5 * s[3]
    return s[0] / den, s[0] / max(s[1], 1), s[1] / den


def pq_prime(acc: np.ndarray, r: float) -> float:
    S, tp, fp, fn = acc.sum(0)[:4]
    return r * S / (r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn)


def r_interval(acc: np.ndarray, lb: float, half: float = 0.005) -> tuple[float, float]:
    """Values of r for which PQ'(r) rounds to the displayed leaderboard score (PQ' increases with r)."""
    S, tp, fp, fn = acc.sum(0)[:4]

    def solve(v):  # PQ'(r) = 2rS / (TP + FP + r(TP + FN)) = v
        return v * (tp + fp) / (2 * S - v * (tp + fn))
    return float(solve(lb - half)), float(solve(lb + half))


def per_image(data, keep_fn) -> np.ndarray:
    rows = []
    for img, f, mats in data:
        keep = keep_fn(img, f)
        acc = np.zeros(5)  # sum IoU of TPs, TP, FP, FN (over the image's annotators), kept instances
        for m in mats:
            sub = m[:, keep]
            s = pq_from_iou(sub, sub.shape[0], int(keep.sum()))
            acc[:4] += [s.tp_iou, s.tp, s.fp, s.fn]
        acc[4] = keep.sum()
        rows.append(acc)
    return np.array(rows)


def evaluate_row(row: dict, runs: str, folds: dict, cache: dict) -> dict:
    path = os.path.join(runs, row["cands"])
    if path not in cache:
        d = pickle.load(open(path, "rb"))
        cache[path] = (d, None)
    d, scores = cache[path]
    data = d["data"]
    sel = row.get("select", "best")
    out = {"name": row["name"], "cands": row["cands"], "select": sel, "base_cfg": str(d["cfg"])}
    if sel == "rule":
        acc = per_image(data, lambda img, f: (np.exp(f[:, 0]) >= 500) & (f[:, 2] >= 0.9) if len(f) else np.zeros(0, bool))
    elif sel == "all":
        acc = per_image(data, lambda img, f: np.ones(len(f), bool))
    else:
        if scores is None:
            scores = crossfit_scores(data, folds)
            cache[path] = (d, scores)

        def keep_at(t):
            return lambda img, f: scores[img] >= t if len(f) else np.zeros(0, bool)
        if sel == "best":
            accs = {t: per_image(data, keep_at(t)) for t in GRID}
            pqs = {t: pq_of(a)[0] for t, a in accs.items()}
            t_best = max(pqs, key=pqs.get)
            acc = accs[t_best]
            out["score_thr"] = float(t_best)
            out["pq_by_thr"] = {f"{t:.3f}": round(float(v), 4) for t, v in pqs.items()}
            # nested: each fold's threshold chosen on the other folds
            fid = np.array([folds[img] for img, _, _ in data])
            nested = np.zeros_like(acc)
            chosen = {}
            for k in np.unique(fid):
                t_k = max(GRID, key=lambda t: pq_of(accs[t][fid != k])[0])
                nested[fid == k] = accs[t_k][fid == k]
                chosen[int(k)] = float(t_k)
            out["nested_PQ"] = float(pq_of(nested)[0])
            out["nested_thr_by_fold"] = chosen
        else:
            acc = per_image(data, keep_at(float(sel)))
            out["score_thr"] = float(sel)
    P, S, R = pq_of(acc)
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(acc), (B, len(acc)))
    boot = np.array([pq_of(acc[i])[0] for i in idx])
    tot = acc.sum(0)
    multi = np.array([len(m) > 1 for _, _, m in data])
    out.update({
        "images": len(acc), "annotator_images": int(sum(len(m) for _, _, m in data)),
        "PQ": float(P), "SQ": float(S), "RQ": float(R),
        "PQ_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
        "TP": int(tot[1]), "FP": int(tot[2]), "FN": int(tot[3]),
        "pred_per_image": float(acc[:, 4].mean()),
        "PQ_prime_r0.67": float(pq_prime(acc, 0.67)),
        "PQ_multi_annotator_images": float(pq_of(acc[multi])[0]) if multi.any() else None,
        "n_multi_annotator_images": int(multi.sum()),
    })
    out["_acc"] = acc
    out["_images"] = [img for img, _, _ in data]
    return out


def paired_diff(a: dict, b: dict) -> dict:
    common = sorted(set(a["_images"]) & set(b["_images"]))
    ia = {img: i for i, img in enumerate(a["_images"])}
    ib = {img: i for i, img in enumerate(b["_images"])}
    A = a["_acc"][[ia[c] for c in common]]
    Bm = b["_acc"][[ib[c] for c in common]]
    rng = np.random.RandomState(1)
    idx = rng.randint(0, len(common), (B, len(common)))
    diffs = np.array([pq_of(A[i])[0] - pq_of(Bm[i])[0] for i in idx])
    return {"ref": b["name"], "common_images": len(common), "delta": float(pq_of(A)[0] - pq_of(Bm)[0]),
            "delta_ci95": np.percentile(diffs, [2.5, 97.5]).tolist(), "p_delta_le_0": float((diffs <= 0).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--runs", default=os.environ.get("FILAMENT_RUNS", "runs"))
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec, encoding="utf-8"))
    folds = dict(pd.read_csv(a.folds).values)
    cache, res = {}, {}
    for row in spec["rows"]:
        res[row["name"]] = evaluate_row(row, a.runs, folds, cache)
        r = res[row["name"]]
        print(f"{r['name']:<48} PQ {r['PQ']:.4f} [{r['PQ_ci95'][0]:.3f}, {r['PQ_ci95'][1]:.3f}]"
              f"  thr {r.get('score_thr', '-')}  nested {r.get('nested_PQ', float('nan')):.4f}  PQ' {r['PQ_prime_r0.67']:.4f}"
              f"  TP/FP/FN {r['TP']}/{r['FP']}/{r['FN']}", flush=True)
    for row in spec["rows"]:
        if row.get("ref"):
            res[row["name"]]["vs_ref"] = paired_diff(res[row["name"]], res[row["ref"]])
            v = res[row["name"]]["vs_ref"]
            print(f"  {row['name']:<46} - {row['ref']:<30} {v['delta']:+.4f} [{v['delta_ci95'][0]:+.4f}, {v['delta_ci95'][1]:+.4f}]")
    subs, accs = [], {}
    for s in spec.get("submissions", []):
        r = res[s["row"]]
        lo, hi = r_interval(r["_acc"], s["lb"])
        accs[s["id"]] = r["_acc"]
        subs.append({"id": s["id"], "row": s["row"], "oof_images": r["images"], "full_oof": r["images"] == len(folds),
                     "oof_PQ": r["PQ"], "lb": s["lb"], "lb_over_oof": s["lb"] / r["PQ"],
                     "PQ_prime_r0.67": r["PQ_prime_r0.67"], "r_lo": lo, "r_hi": hi})
        print(f"{s['id']}: OOF {r['PQ']:.4f} ({r['images']} img)  LB {s['lb']:.2f}  LB/OOF {s['lb'] / r['PQ']:.3f}"
              f"  PQ'(0.67) {r['PQ_prime_r0.67']:.4f}  r in [{lo:.3f}, {hi:.3f}]")
    summary = {}
    full = [s for s in subs if s["full_oof"]]  # submissions whose OOF estimate covers all training images
    if full:
        rs = np.round(np.arange(0.5, 1.0001, 0.001), 3)
        sse = [sum((pq_prime(accs[s["id"]], r) - s["lb"]) ** 2 for s in full) for r in rs]
        r_fit = float(rs[int(np.argmin(sse))])
        for s in subs:
            s["PQ_prime_fit"] = float(pq_prime(accs[s["id"]], r_fit))
        oof, lb = pd.Series([s["oof_PQ"] for s in full]), pd.Series([s["lb"] for s in full])
        summary = {"full_oof_submissions": [s["id"] for s in full],
                   "r_least_squares": r_fit,
                   "max_abs_residual_at_r_fit": float(max(abs(s["PQ_prime_fit"] - s["lb"]) for s in full)),
                   "r_intersection_full": [max(s["r_lo"] for s in full), min(s["r_hi"] for s in full)],
                   "lb_over_oof_range_full": [min(s["lb_over_oof"] for s in full), max(s["lb_over_oof"] for s in full)],
                   "lb_over_oof_mean_full": float(np.mean([s["lb_over_oof"] for s in full])),
                   "spearman_oof_lb_full": float(oof.corr(lb, method="spearman")),
                   "pearson_oof_lb_full": float(oof.corr(lb))}
        print(json.dumps(summary, indent=1))
        for s in subs:
            print(f"  {s['id']}: LB {s['lb']:.2f}  PQ'(r_fit={r_fit:.3f}) {s['PQ_prime_fit']:.4f}")
    for r in res.values():
        r.pop("_acc")
        r.pop("_images")
    json.dump({"rows": list(res.values()), "submissions": subs, "summary": summary,
               "bootstrap": B, "grid": GRID.tolist()}, open(a.out, "w"), indent=1)
    print("->", a.out)
