"""Train one U-Net run and record timing / GPU memory.

Default config (the smoke test and later the main run): MobileNetV2 encoder without ImageNet
weights, VV+VH+ratio input, BCE+Dice loss, Adam lr 1e-3, batch 8, mixed precision.

Examples (from the project root, with tf-gpu-env active):
    python -m src.train --epochs 5 --out results/smoke          # smoke test
    python -m src.train --epochs 60 --seed 0 --out results/R01

Writes to --out: best.keras (best val water IoU), history.csv, config.json, timing.json.
"""
import argparse
import json
import time

import numpy as np
import tensorflow as tf

from src.data import ROOT
from src.losses import WaterIoU, bce_dice_loss, bce_loss, dice_loss
from src.models import ENCODERS, build_unet
from src.pipeline import CHANNEL_SETS, load_arrays, make_eval_dataset, make_train_dataset

LOSSES = {"bce": bce_loss, "dice": dice_loss, "bce_dice": bce_dice_loss}


class EpochTimer(tf.keras.callbacks.Callback):
    """Seconds per epoch (training + validation)."""

    def on_train_begin(self, logs=None):
        self.seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(time.perf_counter() - self._t0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="mobilenetv2", choices=ENCODERS)
    p.add_argument("--channels", default="vv_vh_ratio", choices=list(CHANNEL_SETS))
    p.add_argument("--loss", default="bce_dice", choices=list(LOSSES))
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument("--out", required=True, help="output folder, relative to the project root")
    return p.parse_args()


def main():
    args = parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)
    if not args.no_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus, "| policy:", tf.keras.mixed_precision.global_policy().name)

    X, Y = load_arrays("train", args.channels)
    Xv, Yv = load_arrays("val", args.channels)
    train_ds = make_train_dataset(X, Y, batch=args.batch, seed=args.seed)
    val_ds = make_eval_dataset(Xv, Yv, batch=args.batch)

    model = build_unet(args.encoder, X.shape[-1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=LOSSES[args.loss],
        metrics=[WaterIoU()],
    )
    print(f"{model.name}: {model.count_params():,} parameters")

    timer = EpochTimer()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            out / "best.keras", monitor="val_water_iou", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(out / "history.csv"),
        timer,
    ]

    if gpus:
        tf.config.experimental.reset_memory_stats("GPU:0")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
    peak_mb = (
        tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1024**2 if gpus else None
    )

    sec = timer.seconds
    timing = {
        "sec_per_epoch": [round(s, 1) for s in sec],
        "mean_sec_per_epoch": round(float(np.mean(sec)), 1),
        # epoch 1 includes graph building / cuDNN autotuning, so also report without it
        "mean_sec_per_epoch_excl_first": round(float(np.mean(sec[1:])), 1) if len(sec) > 1 else None,
        "peak_gpu_memory_mb": round(peak_mb, 0) if peak_mb is not None else None,
        "parameters": model.count_params(),
    }
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
