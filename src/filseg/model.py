"""Network definition, input normalisation and test-time-augmented inference (shared by the
training kernel, the inference notebook and the submission builder)."""
from __future__ import annotations

import cv2
import numpy as np
import torch

NATIVE = 2048


def build_model(encoder: str = "resnet34", arch: str = "Unet", pretrained: bool = False):
    import segmentation_models_pytorch as smp
    return getattr(smp, arch)(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=1)


def load_weights(model, path: str, device: str = "cpu"):
    sd = torch.load(path, map_location=device)
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    return model.to(device).eval()


def load_image(path: str, size: int) -> np.ndarray:
    """Grayscale H-alpha JPEG -> uint8 (size, size) with area interpolation."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def disk_stats(img: np.ndarray) -> tuple[float, float]:
    """Robust location/scale of the solar disk (pixels > 20): median and IQR / 1.349."""
    x = img[img > 20].astype(np.float32)
    if x.size < 1000:
        x = img.astype(np.float32).ravel()
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(max((q3 - q1) / 1.349, 1.0))


def normalise_t(x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    """x: (B,1,H,W) raw 0..255, stats: (B,2) -> per-image standardised disk / 4, off-disk = -2."""
    med = stats[:, 0].view(-1, 1, 1, 1)
    sc = stats[:, 1].view(-1, 1, 1, 1)
    z = ((x - med) / sc).clamp(-8, 8)
    z = torch.where(x > 20, z, torch.full_like(z, -8))
    return z / 4.0


@torch.no_grad()
def predict_tta(models, img: np.ndarray, device: str = "cpu", transposed: bool = False) -> np.ndarray:
    """Average sigmoid probability over models and the 4 flips; img is uint8 at model resolution.
    transposed=True uses the 4 flips of the transposed image (the other half of the 8 dihedral views)."""
    x = torch.from_numpy(img).float()[None, None].to(device)
    st = torch.tensor([disk_stats(img)], dtype=torch.float32, device=device)
    x = normalise_t(x, st)
    if transposed:
        x = x.transpose(2, 3)
    acc = torch.zeros_like(x)
    use_amp = device.startswith("cuda")
    for m in models:
        for fh in (False, True):
            for fv in (False, True):
                xx = x.flip(3) if fh else x
                xx = xx.flip(2) if fv else xx
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    p = torch.sigmoid(m(xx).float())
                p = p.flip(2) if fv else p
                p = p.flip(3) if fh else p
                acc += p
    if transposed:
        acc = acc.transpose(2, 3)
    return (acc / (4 * len(models)))[0, 0].cpu().numpy()
