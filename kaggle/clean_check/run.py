"""Clean-environment check of the public repository (inference only, no training).

1. copy the repo (attached private dataset `filament-repo-check` = byte copy of this repository, weights included,
   plus the submitted s10 CSV as reference) to /tmp, as a `git clone` would;
2. create a fresh virtualenv and `pip install -r requirements.txt` (the notebook does not see the Kaggle image's
   packages; only ipykernel is added so that Jupyter can run it);
3. execute notebooks/filament_pipeline.ipynb end-to-end with its DEFAULT configuration (configs/final_all.json =
   submission s10) on 1 in STEP test images (the full 180 take ~1 h on a T4; the subset keeps GPU use small);
4. compare its submission.csv with the submitted s10 CSV on the same images (instance counts, Hungarian-matched
   instance IoU, pixel IoU of the union of masks, identical RLEs).
Outputs in /kaggle/working: filament_pipeline_executed.ipynb, submission_subset.csv, compare.json,
compare_per_image.csv, cell_times.json, run_summary.json, env_probe.json, pip_install.log, pip_freeze.txt.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

STEP = 5                 # 1 = all 180 test images
MAX_TRAIN_IMAGES = "4"   # notebook section 6 (in-sample PQ demo); default there is 20
T0 = time.time()
OUT = "/kaggle/working"
WORK = "/tmp/filament_check"
REPO = f"{WORK}/repo"
VENV = f"{WORK}/venv"
PY = f"{VENV}/bin/python"
REF_NAME = "s10_all_d8_2048tta_g10_resc325.csv"
SUMMARY = {"step": STEP}
os.makedirs(WORK, exist_ok=True)
ENV = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PIP_TARGET")}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def run(cmd, logname=None, check=True, env=None, tail=12):
    log("$", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env or ENV)
    if logname:
        with open(os.path.join(OUT, logname), "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n{r.stdout}")
    for line in r.stdout.strip().splitlines()[-tail:]:
        print("   |", line, flush=True)
    if check and r.returncode:
        finish(f"failed: {' '.join(cmd[:4])}... (exit {r.returncode})")
        raise SystemExit(1)
    return r


def finish(status):
    SUMMARY["status"] = status
    SUMMARY["elapsed_s"] = round(time.time() - T0)
    with open(f"{OUT}/run_summary.json", "w") as f:
        json.dump(SUMMARY, f, indent=1)
    log("STATUS", status)


# ---------------------------------------------------------------- 1. repo + data
log("inputs:", sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*"))[:20])
hits = glob.glob("/kaggle/input/**/configs/final_all.json", recursive=True)
if hits:
    src = os.path.dirname(os.path.dirname(hits[0]))
else:  # dataset kept the zipped folder
    z = glob.glob("/kaggle/input/**/repo.zip", recursive=True)[0]
    zipfile.ZipFile(z).extractall(f"{WORK}/repo_zip")
    src = os.path.dirname(os.path.dirname(glob.glob(f"{WORK}/repo_zip/**/configs/final_all.json", recursive=True)[0]))
shutil.copytree(src, REPO, ignore=shutil.ignore_patterns("__pycache__"))
SUMMARY["repo_src"] = src
log("repo copied from", src, "->", REPO, sorted(os.listdir(REPO)))
ref = glob.glob(f"/kaggle/input/**/{REF_NAME}", recursive=True)[0]
data_real = [d for d in glob.glob("/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026", recursive=True) if os.path.isdir(f"{d}/test/test_images")][0]
names = sorted(f for f in os.listdir(f"{data_real}/test/test_images") if f.endswith(".jpeg"))
subset = names[::STEP]
SUB = f"{WORK}/data/MAGFiLO_1.0_Kaggle_2026"
os.makedirs(f"{SUB}/test/test_images")
for n in subset:
    os.symlink(f"{data_real}/test/test_images/{n}", f"{SUB}/test/test_images/{n}")
os.symlink(f"{data_real}/train", f"{SUB}/train")
with open(f"{WORK}/subset.txt", "w") as f:
    f.write("\n".join(n.rsplit(".", 1)[0] for n in subset))
SUMMARY.update(data=data_real, n_test=len(names), n_subset=len(subset))
log(f"data {data_real}: {len(names)} test images, subset {len(subset)} (1 in {STEP})")

# ---------------------------------------------------------------- 2. clean virtualenv from requirements.txt
run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=False)
r = run([sys.executable, "-m", "venv", VENV], check=False)
if r.returncode or run([PY, "-m", "pip", "--version"], check=False).returncode:
    shutil.rmtree(VENV, ignore_errors=True)  # Debian python without ensurepip -> uv creates the venv
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run([sys.executable, "-m", "uv", "venv", "--seed", "--python", sys.executable, VENV])
    SUMMARY["venv"] = "uv venv --seed"
else:
    SUMMARY["venv"] = "python -m venv"
t = time.time()
run([PY, "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off", "-r", f"{REPO}/requirements.txt"], "pip_install.log")
SUMMARY["pip_requirements_s"] = round(time.time() - t)
run([PY, "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off", "ipykernel"], "pip_install.log")
run([PY, "-m", "ipykernel", "install", "--user", "--name", "filament-venv"])

PROBE = r"""
import json, sys
from importlib.metadata import version
d = {p: version(p) for p in ["numpy", "pandas", "scipy", "opencv-python-headless", "pycocotools", "scikit-learn",
                             "torch", "torchvision", "segmentation-models-pytorch", "timm", "matplotlib"]}
import torch, timm, segmentation_models_pytorch, cv2, sklearn, pycocotools.mask
d["python"] = sys.version.split()[0]
d["torch_file"] = torch.__file__
d["torch_cuda"] = torch.version.cuda
d["cuda_available"] = torch.cuda.is_available()
try:
    x = torch.randn(1, 1, 64, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        y = torch.nn.functional.conv2d(x, torch.randn(1, 1, 3, 3, device="cuda"))
    torch.cuda.synchronize()
    d["cuda_smoke"] = "ok " + torch.cuda.get_device_name(0)
except Exception as e:
    d["cuda_smoke"] = "FAIL " + repr(e)[:300]
print("PROBE " + json.dumps(d))
"""


def probe():
    r = run([PY, "-c", PROBE], check=False, tail=3)
    line = [l for l in r.stdout.splitlines() if l.startswith("PROBE ")]
    return json.loads(line[-1][6:]) if line else {"cuda_smoke": "FAIL import: " + r.stdout[-500:]}


env = probe()
SUMMARY["probe_requirements"] = env
if not env.get("cuda_smoke", "").startswith("ok"):
    # the default PyPI torch wheel may need a newer driver than the host has: try official CUDA builds of the
    # same versions (documented in the README if this path is taken)
    for idx in ("cu128", "cu126"):
        log("CUDA not usable with the default wheel; trying", idx)
        run([PY, "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off", f"torch==2.15.0+{idx}",
             f"torchvision==0.29.2+{idx}", "--index-url", f"https://download.pytorch.org/whl/{idx}",
             "--extra-index-url", "https://pypi.org/simple"], "pip_install.log", check=False)
        env = probe()
        SUMMARY[f"probe_{idx}"] = env
        if env.get("cuda_smoke", "").startswith("ok"):
            break
with open(f"{OUT}/env_probe.json", "w") as f:
    json.dump(env, f, indent=1)
with open(f"{OUT}/pip_freeze.txt", "w") as f:
    f.write(run([PY, "-m", "pip", "freeze"], check=False, tail=0).stdout)
if not env.get("cuda_smoke", "").startswith("ok"):
    finish("no usable CUDA in the clean venv: not running the notebook on CPU")
    raise SystemExit(1)

# ---------------------------------------------------------------- 3. run the notebook (default config)
EXEC = r"""
import json, sys
from datetime import datetime
import nbformat
from nbclient import NotebookClient
src, dst, cwd, times = sys.argv[1:5]
nb = nbformat.read(src, as_version=4)
err = None
try:
    NotebookClient(nb, timeout=None, kernel_name="filament-venv", resources={"metadata": {"path": cwd}}).execute()
except Exception as e:
    err = repr(e)[-4000:]
nbformat.write(nb, dst)
tt = []
for i, c in enumerate(nb.cells):
    ex = c.get("metadata", {}).get("execution", {})
    if "iopub.execute_input" in ex and "shell.execute_reply" in ex:
        a, b = (datetime.fromisoformat(ex[k].replace("Z", "+00:00")) for k in ("iopub.execute_input", "shell.execute_reply"))
        tt.append({"cell": i, "seconds": round((b - a).total_seconds(), 1), "first_line": c.source.splitlines()[0][:90]})
json.dump(tt, open(times, "w"), indent=1)
for t in tt:
    print(t)
if err:
    print("NOTEBOOK ERROR", err)
    sys.exit(1)
"""
with open(f"{WORK}/exec_nb.py", "w") as f:
    f.write(EXEC)
nb_env = dict(ENV, FILAMENT_DATA=SUB, MAX_TRAIN_IMAGES=MAX_TRAIN_IMAGES)
nb_env.pop("FILAMENT_CONFIG", None)  # test the notebook's own default
t = time.time()
r = run([sys.executable, f"{WORK}/exec_nb.py", f"{REPO}/notebooks/filament_pipeline.ipynb",
         f"{OUT}/filament_pipeline_executed.ipynb", f"{REPO}/notebooks", f"{OUT}/cell_times.json"],
        "notebook_run.log", check=False, env=nb_env, tail=40)
SUMMARY["notebook_s"] = round(time.time() - t)
SUMMARY["notebook_ok"] = r.returncode == 0
if os.path.exists(f"{REPO}/notebooks/submission.csv"):
    shutil.copy(f"{REPO}/notebooks/submission.csv", f"{OUT}/submission_subset.csv")
if r.returncode:
    finish("notebook failed (see filament_pipeline_executed.ipynb / notebook_run.log)")
    raise SystemExit(1)

# ---------------------------------------------------------------- 4. compare with the submitted s10
COMPARE = r"""
import json, sys
import numpy as np, pandas as pd
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment
sub_path, ref_path, subset_path, out = sys.argv[1:5]
imgs = open(subset_path).read().split()
def by_image(path):
    df = pd.read_csv(path)
    d = {}
    for fid, c in zip(df.filament_id, df.segmentation_rle):
        d.setdefault(fid.rsplit("_", 1)[0], []).append(c)
    return d
A, B = by_image(sub_path), by_image(ref_path)
assert set(A) <= set(imgs), "notebook wrote images outside the subset"
rows, ious = [], []
for im in imgs:
    a = [{"size": [2048, 2048], "counts": c.encode()} for c in A.get(im, [])]
    b = [{"size": [2048, 2048], "counts": c.encode()} for c in B.get(im, [])]
    r = {"image": im, "n_kernel": len(a), "n_s10": len(b), "identical_rle": len(set(A.get(im, [])) & set(B.get(im, [])))}
    sa = float(mu.area(mu.merge(a))) if a else 0.0
    sb = float(mu.area(mu.merge(b))) if b else 0.0
    if a and b:
        u = float(mu.iou([mu.merge(a)], [mu.merge(b)], [0])[0, 0])
        inter = u * (sa + sb) / (1 + u)
        M = np.asarray(mu.iou(a, b, [0] * len(b)))
        ri, ci = linear_sum_assignment(-M)
        m = M[ri, ci]
        tp = m > 0.5
        ious += list(m[tp])
        r.update(matched=int(tp.sum()), matched_iou_mean=float(m[tp].mean()) if tp.any() else None,
                 pq_vs_s10=float(m[tp].sum() / (tp.sum() + 0.5 * (len(a) - tp.sum()) + 0.5 * (len(b) - tp.sum()))))
        extra = [float(mu.area(a[i])) for i in range(len(a)) if i not in set(ri[tp])]
        miss = [float(mu.area(b[j])) for j in range(len(b)) if j not in set(ci[tp])]
        r.update(unmatched_kernel_areas=extra, unmatched_s10_areas=miss)
    else:
        inter = 0.0
        r.update(matched=0, matched_iou_mean=None, pq_vs_s10=1.0 if not a and not b else 0.0,
                 unmatched_kernel_areas=[float(mu.area(x)) for x in a], unmatched_s10_areas=[float(mu.area(x)) for x in b])
    r.update(pixels_kernel=sa, pixels_s10=sb, pixel_iou=inter / (sa + sb - inter) if sa + sb - inter > 0 else 1.0, _inter=inter)
    rows.append(r)
df = pd.DataFrame(rows)
I, U = df._inter.sum(), (df.pixels_kernel + df.pixels_s10 - df._inter).sum()
ious = np.array(ious)
tp = int(df.matched.sum())
res = {
    "images": len(df), "filaments_kernel": int(df.n_kernel.sum()), "filaments_s10": int(df.n_s10.sum()),
    "images_same_count": int((df.n_kernel == df.n_s10).sum()),
    "matched_iou_gt_0.5": tp, "unmatched_kernel": int(df.n_kernel.sum()) - tp, "unmatched_s10": int(df.n_s10.sum()) - tp,
    "matched_iou_mean": float(ious.mean()) if len(ious) else None, "matched_iou_min": float(ious.min()) if len(ious) else None,
    "matched_iou_ge_0.99": int((ious >= 0.99).sum()), "matched_iou_ge_0.95": int((ious >= 0.95).sum()),
    "identical_rle": int(df.identical_rle.sum()),
    "pq_kernel_vs_s10": float(ious.sum() / (tp + 0.5 * (df.n_kernel.sum() - tp) + 0.5 * (df.n_s10.sum() - tp))),
    "pixel_iou_global": float(I / U), "pixel_iou_min_image": float(df.pixel_iou.min()),
}
df.drop(columns="_inter").to_csv(out + "/compare_per_image.csv", index=False)
json.dump(res, open(out + "/compare.json", "w"), indent=1)
print(json.dumps(res, indent=1))
"""
with open(f"{WORK}/compare.py", "w") as f:
    f.write(COMPARE)
r = run([PY, f"{WORK}/compare.py", f"{OUT}/submission_subset.csv", ref, f"{WORK}/subset.txt", OUT], check=False, tail=30)
SUMMARY["compare_ok"] = r.returncode == 0
if r.returncode == 0:
    SUMMARY["compare"] = json.load(open(f"{OUT}/compare.json"))
finish("ok" if r.returncode == 0 else "compare failed")
