"""Tests: filseg.metric equals a literal numpy port of the organisers' self-evaluation notebook, and the
post-processing returns disjoint instances.

    python -m pytest tests            # or: python tests/test_metric_matches_official.py

The synthetic tests need no data. test_pq_matches_official_on_training_annotations uses the training annotations
($FILAMENT_GT_CSV from tools/build_gt.py, or the COCO file under $FILAMENT_DATA) and is skipped when they are absent.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from filseg.data import build_gt_table, counts_to_mask, load_coco, mask_to_counts  # noqa: E402
from filseg.metric import GTCache, PQStats  # noqa: E402
from filseg.paths import GT_CSV, TRAIN_JSON  # noqa: E402
from filseg.postproc import PostCfg, instances  # noqa: E402

N = 2048


# ---- literal port of the official notebook (torch -> numpy float32) -------------------------------------------
def _rles_to_layers(rles):
    if not rles:
        return np.zeros((0, N, N), np.float32)
    m = mask_util.decode([{"size": [N, N], "counts": r} for r in rles])
    return m.transpose(2, 0, 1).astype(np.float32)


def official_pq(gt_df, pred_df):
    """gt_df.filament_id = '<annotator>-<image>_<k>', pred_df.filament_id = '<image>_<k>'."""
    gti = gt_df.filament_id.str.split("_", n=1).str[0]
    pri = pred_df.filament_id.str.split("_", n=1).str[0]
    tp_iou, fp, fn = [], 0, 0
    for ai in gti.unique():
        image_id = ai.split("-", maxsplit=1)[1]
        G = _rles_to_layers(gt_df.loc[gti == ai, "segmentation_rle"].tolist())
        P = _rles_to_layers(pred_df.loc[pri == image_id, "segmentation_rle"].tolist())
        ng, npd = len(G), len(P)
        if ng == 0:
            fp += npd
            continue
        if npd == 0:
            fn += ng
            continue
        gf, pf = G.reshape(ng, -1), P.reshape(npd, -1)
        inter = gf @ pf.T
        union = gf.sum(1)[:, None] + pf.sum(1)[None] - inter
        iou = np.where(union == 0, 0.0, inter / union)
        hit = iou > 0.5
        tp_iou += iou[hit].tolist()
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    return sum(tp_iou) / (len(tp_iou) + 0.5 * fp + 0.5 * fn), len(tp_iou), fp, fn


# ---- helpers ------------------------------------------------------------------------------------------------------
def _perturbed_predictions(gt, imgs, seed=0):
    """First annotator's masks dilated / eroded / shifted, plus a random blob, made disjoint (as the LB requires)."""
    rng = np.random.RandomState(seed)
    rows = []
    for img in imgs:
        g = gt[gt.image == img]
        first = g.annotator_image.iloc[0]
        lab = np.zeros((N, N), np.int32)
        k = 0
        for c in g[g.annotator_image == first].segmentation_rle:
            m = mask_util.decode({"size": [N, N], "counts": c.encode()})
            op = rng.randint(3)
            if op == 0:
                m = cv2.dilate(m, np.ones((7, 7), np.uint8))
            elif op == 1:
                m = cv2.erode(m, np.ones((5, 5), np.uint8))
            else:
                m = np.roll(m, rng.randint(-15, 15), axis=1)
            m = m.astype(bool) & (lab == 0)
            if m.sum() == 0:
                continue
            k += 1
            lab[m] = k
        yy, xx = np.ogrid[:N, :N]
        blob = ((yy - 1000) ** 2 + (xx - 1000) ** 2 < 900) & (lab == 0)
        k += 1
        lab[blob] = k
        rows += [{"filament_id": f"{img}_{j}", "segmentation_rle": mask_to_counts(lab == j)} for j in range(1, k + 1)]
    return pd.DataFrame(rows)


def _filseg_pq(gt, imgs, pred_df):
    cache, tot = GTCache(gt.to_dict("records")), PQStats()
    for img in imgs:
        pr = pred_df[pred_df.filament_id.str.split("_").str[0] == img]
        lab = np.zeros((N, N), np.int32)
        for j, c in enumerate(pr.segmentation_rle, 1):
            lab[counts_to_mask(c) > 0] = j
        for s in cache.score(img, lab, len(pr)).values():
            tot.add(s)
    return tot


def _assert_same(gt, imgs):
    pred_df = _perturbed_predictions(gt, imgs)
    ref = official_pq(gt[["filament_id", "segmentation_rle"]], pred_df)
    tot = _filseg_pq(gt, imgs, pred_df)
    print("official port: PQ %.6f TP %d FP %d FN %d" % ref)
    print("filseg       : PQ %.6f TP %d FP %d FN %d" % (tot.pq, tot.tp, tot.fp, tot.fn))
    assert abs(ref[0] - tot.pq) < 1e-6 and (ref[1], ref[2], ref[3]) == (tot.tp, tot.fp, tot.fn)


def _synthetic_gt(seed=1):
    """Two images x two annotators of random disjoint ellipses (the second annotator jitters and drops some)."""
    rng = np.random.RandomState(seed)
    rows = []
    for img in ("20990101000000Bh", "20990102000000Uh"):
        base = [(rng.randint(200, 1800), rng.randint(200, 1800), rng.randint(15, 60), rng.randint(4, 15), rng.randint(180))
                for _ in range(8)]
        for a in ("A1", "A2"):
            lab = np.zeros((N, N), np.uint8)
            k = 0
            for cy, cx, ax1, ax2, ang in base:
                if a == "A2" and rng.rand() < 0.25:
                    continue
                m = np.zeros((N, N), np.uint8)
                j = rng.randint(-3, 4) if a == "A2" else 0
                cv2.ellipse(m, (cx + j, cy - j), (ax1, ax2), ang, 0, 360, 1, -1)
                m = (m > 0) & (lab == 0)
                if not m.any():
                    continue
                k += 1
                lab[m] = 1
                rows.append({"filament_id": f"{a}-{img}_{k}", "annotator_image": f"{a}-{img}", "image": img,
                             "segmentation_rle": mask_to_counts(m)})
    return pd.DataFrame(rows)


# ---- tests --------------------------------------------------------------------------------------------------------
def test_pq_matches_official_synthetic():
    gt = _synthetic_gt()
    _assert_same(gt, gt.image.unique())


def test_pq_matches_official_on_training_annotations():
    if os.path.exists(GT_CSV):
        gt = pd.read_csv(GT_CSV)
    elif os.path.exists(TRAIN_JSON):
        gt = pd.DataFrame(build_gt_table(load_coco(TRAIN_JSON)))
    else:
        import pytest
        pytest.skip("training annotations not found (set FILAMENT_DATA or FILAMENT_GT_CSV)")
    multi = gt.groupby("image").annotator_image.nunique()
    imgs = gt[gt.image.isin(multi[multi > 1].index)].image.unique()[:6]
    _assert_same(gt[gt.image.isin(imgs)], imgs)


def test_rle_roundtrip():
    m = np.zeros((N, N), np.uint8)
    cv2.ellipse(m, (700, 900), (120, 20), 30, 0, 360, 1, -1)
    assert np.array_equal(counts_to_mask(mask_to_counts(m)), m)


def test_instances_are_disjoint_filled_and_grouped():
    p = np.zeros((N, N), np.float32)
    p[1000:1010, 900:1000] = 0.9        # two fragments of one filament, 12 px apart ...
    p[1000:1010, 1012:1100] = 0.8
    p[400:440, 400:440] = 0.9           # ... a separate blob with a hole
    p[415:425, 415:425] = 0.1
    lab0, n0 = instances(p, PostCfg(thr=0.5, seed_thr=0.0, group_r=0))
    lab1, n1 = instances(p, PostCfg(thr=0.5, seed_thr=0.0, group_r=10))
    assert (n0, n1) == (3, 2)           # grouping joins the two fragments without adding pixels
    assert ((lab1 > 0) == (lab0 > 0)).all()
    assert lab1[420, 420] > 0           # hole filled
    assert lab1.dtype == np.int32 and lab1.max() == n1   # one label per pixel: instances cannot overlap


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # pytest.skip raises a subclass of BaseException, handled below
                print("FAIL", name, repr(e))
                raise
            except BaseException as e:
                print("SKIP", name, e)
