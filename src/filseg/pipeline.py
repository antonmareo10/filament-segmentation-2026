"""End-to-end inference: H-alpha JPEG -> non-overlapping filament masks -> submission rows.

The configuration (JSON) lists the ensemble members and the post-processing:

{
  "models": [{"weights": "weights/r34_1536_f0.pth", "encoder": "resnet34", "arch": "Unet", "size": 1536}, ...],
  "base_cfg": {"thr": 0.5, "seed_thr": 0.0, "min_area": 0},
  "rescorer": "weights/rescorer.pkl",      # optional instance re-scorer (tools/rescore_fit.py)
  "score_thr": 0.35
}

Each model's flip-TTA probability is rounded to 1/255 (as the stored out-of-fold maps used for
tuning), models of one resolution are averaged, resolutions are bilinearly resized to 2048 and
averaged, then `postproc.instances` + the optional re-scorer produce the instances.
"""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch

from .data import mask_to_counts
from .model import build_model, load_image, load_weights, predict_tta
from .postproc import PostCfg, instances, upsample
from .rescore import component_features


class Pipeline:
    def __init__(self, config_path: str, device: str | None = None):
        root = os.path.dirname(os.path.abspath(config_path))
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.groups = defaultdict(list)  # size -> [models]
        self.members = []  # (net, [views]); a view is (input size, transposed flag)
        for m in self.cfg["models"]:
            net = load_weights(build_model(m["encoder"], m.get("arch", "Unet")), os.path.join(root, m["weights"]), self.device)
            self.groups[int(m["size"])].append(net)
            views = [(int(v[0]), bool(v[1])) for v in m.get("views", [[m["size"], False]])]
            self.members.append((net, views))
        self.base = PostCfg(**self.cfg["base_cfg"])
        self.rescorer = None
        if self.cfg.get("rescorer"):
            with open(os.path.join(root, self.cfg["rescorer"]), "rb") as f:
                self.rescorer = pickle.load(f)["model"]
        self.score_thr = float(self.cfg.get("score_thr", 0.0))

    def probability(self, image_path: str) -> np.ndarray:
        """Ensemble probability at native 2048 x 2048 resolution. Every (model, view) map is rounded to 1/255
        (as the stored maps used for tuning), resized to 2048 and averaged with equal weight."""
        if any(len(v) > 1 or v[0][1] for _, v in self.members):
            acc, n, imgs = 0.0, 0, {}
            for net, views in self.members:
                for size, tr in views:
                    if size not in imgs:
                        imgs[size] = load_image(image_path, size)
                    p = np.round(predict_tta([net], imgs[size], self.device, transposed=tr) * 255) / 255.0
                    acc = acc + upsample(p.astype(np.float32))
                    n += 1
            return (acc / n).astype(np.float32)
        maps = []
        for size, nets in self.groups.items():
            img = load_image(image_path, size)
            per_model = [np.round(predict_tta([n], img, self.device) * 255) / 255.0 for n in nets]
            maps.append((np.mean(per_model, axis=0).astype(np.float32), len(nets)))
        if len(maps) == 1:
            return upsample(maps[0][0])
        ups = [upsample(m) for m, _ in maps]
        w = [k for _, k in maps]
        return sum(wi * u for wi, u in zip(w, ups)) / sum(w)

    def segment(self, prob: np.ndarray) -> tuple[np.ndarray, int]:
        lab, n = instances(prob, self.base)
        if self.rescorer is not None and n:
            keep = self.rescorer.predict(component_features(prob, lab, n)) >= self.score_thr
            remap = np.zeros(n + 1, np.int32)
            remap[1:][keep] = np.arange(1, keep.sum() + 1)
            lab, n = remap[lab], int(keep.sum())
        return lab, n

    def rows(self, image_path: str) -> list[dict]:
        name = os.path.basename(image_path).rsplit(".", 1)[0]
        lab, n = self.segment(self.probability(image_path))
        return [{"filament_id": f"{name}_{k}", "segmentation_rle": mask_to_counts(lab == k)} for k in range(1, n + 1)]
