Exact training scripts of the released models (Kaggle, GPU T4 x2):

* `run_r34_1536.py` — ResNet-34 U-Net at 1536 px (folds were run in three kernels: `FOLDS=0`, `1,2`, `3,4`).
* `run_effv2s_1536.py` — EfficientNetV2-S U-Net at 1536 px (same split into kernels).

`../train_unet/run.py` is the maintained version of the same script: identical data, normalisation, target,
model, loss, schedule and augmentation; it additionally stores images as uint8 (lower RAM), supports random-crop
training, and has a watchdog that aborts a deadlocked multi-GPU session.
