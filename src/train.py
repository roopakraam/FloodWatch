"""Train one U-Net run.

A run is fully described by a config (src/config.py). Training writes to results/<run_id>/:

    config.json    every setting the run used + provenance (git commit, library versions, GPU)
    history.csv    one row per epoch: losses, water IoU, learning rate, seconds
    train.log      the per-epoch lines printed during training
    best.keras     the model at its best validation water IoU (inference-ready; not tracked by git)
    metrics.json   training summary (epochs, timing, GPU memory, parameters) + validation scores;
                   src/evaluate.py later adds the test and Bolivia scores, once
    per_chip.csv   per-chip validation scores (test / Bolivia rows are added by src/evaluate.py)

Checkpointing, learning-rate reduction and early stopping all watch the validation water IoU and
nothing else. The test and Bolivia splits are never read here.

Run from the project root (tf-gpu-env active):
    python -m src.train --config configs/R04.json
    python -m src.train --config configs/R01.json --set epochs=5 --out results/smoke     # ad-hoc variant
The notebooks call `train_run` directly instead.
"""
if __package__ in (None, ""):  # allow `python src/train.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import gc
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from src.config import FROZEN_PATHS, load_config, resolve, run_dir
from src.data import ROOT
from src.evaluate import evaluate_split, write_json
from src.losses import LOSSES, WaterIoU
from src.models import build_unet, count_parameters
from src.pipeline import load_arrays, make_eval_dataset, make_train_dataset
from src.preprocess import n_channels
from src.runlog import format_epoch, header_line, is_trained, replay, summary_line

MONITOR = "val_water_iou"


def setup_runtime(mixed_precision=True):
    """Process-wide TensorFlow settings. Safe to call more than once."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:  # take GPU memory as needed instead of all 4 GB up front
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # the GPU is already initialised in this process
    keras.mixed_precision.set_global_policy("mixed_float16" if mixed_precision else "float32")
    return gpus


def provenance():
    """Where a result came from: code version, library versions, hardware."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    gpus = tf.config.list_physical_devices("GPU")
    gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name") if gpus else None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        # True if any file that defines training or scoring differs from that commit
        "code_modified": bool(git("status", "--porcelain", "--", *FROZEN_PATHS)),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "gpu": gpu_name,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class EpochLogger(keras.callbacks.Callback):
    """Times every epoch (training + validation, before any checkpoint is written), adds the learning
    rate, the seconds and an is-best flag to the logs, and prints / stores one line per epoch."""

    def __init__(self, total_epochs, log, log_path):
        super().__init__()
        self.total_epochs, self.log, self.log_path = total_epochs, log, log_path
        self.best = -np.inf

    def on_epoch_begin(self, epoch, logs=None):
        self._lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs["seconds"] = round(time.perf_counter() - self._t0, 2)
        logs["learning_rate"] = self._lr
        logs["is_best"] = int(logs[MONITOR] > self.best)
        self.best = max(self.best, logs[MONITOR])
        line = format_epoch({"epoch": epoch, **logs}, self.total_epochs)
        self.log(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")


def train_run(cfg, out_dir=None, force=False, log=print):
    """Train the run described by `cfg`. Returns the metrics dict (training summary + validation scores).

    If the run folder already holds a finished training it is NOT repeated (pass force=True to retrain).
    """
    run_id = cfg.get("run_id") or "adhoc"
    out = Path(out_dir) if out_dir is not None else run_dir(run_id)
    if is_trained(out) and not force:
        return replay(out, log, scores=())

    out.mkdir(parents=True, exist_ok=True)
    for stale in ("history.csv", "train.log", "metrics.json", "per_chip.csv", "best.keras", "best.weights.h5"):
        (out / stale).unlink(missing_ok=True)

    # ---- reproducibility + runtime -----------------------------------------------------------------
    keras.backend.clear_session()
    gc.collect()
    gpus = setup_runtime(cfg["mixed_precision"])
    keras.utils.set_random_seed(cfg["seed"])  # Python, NumPy and TensorFlow seeds
    write_json(out / "config.json", {**cfg, "provenance": provenance()})

    # ---- data --------------------------------------------------------------------------------------
    X, Y = load_arrays("train", cfg["channels"])
    Xv, Yv = load_arrays("val", cfg["channels"])
    train_ds = make_train_dataset(X, Y, batch=cfg["batch_size"], crops_per_chip=cfg["crops_per_chip"],
                                  crop=cfg["crop_size"], seed=cfg["seed"])
    val_ds = make_eval_dataset(Xv, Yv, batch=cfg["eval_batch_size"])

    # ---- model -------------------------------------------------------------------------------------
    if gpus:
        tf.config.experimental.reset_memory_stats("GPU:0")
    model = build_unet(cfg["encoder"], n_channels(cfg["channels"]), bn_momentum=cfg["bn_momentum"])
    model.compile(
        optimizer=keras.optimizers.Adam(cfg["learning_rate"]),
        loss=LOSSES[cfg["loss"]],
        metrics=[WaterIoU(threshold=cfg["threshold"])],
        jit_compile=cfg["jit_compile"],
    )
    params = count_parameters(model)
    log(header_line(cfg, params["total_params"]))

    weights = out / "best.weights.h5"
    callbacks = [
        EpochLogger(cfg["epochs"], log, out / "train.log"),
        keras.callbacks.ModelCheckpoint(weights, monitor=MONITOR, mode="max", save_best_only=True,
                                        save_weights_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, mode="max", factor=cfg["reduce_lr_factor"],
                                          patience=cfg["reduce_lr_patience"], min_lr=cfg["min_learning_rate"]),
        keras.callbacks.EarlyStopping(monitor=MONITOR, mode="max", patience=cfg["early_stopping_patience"],
                                      restore_best_weights=True),
        keras.callbacks.CSVLogger(out / "history.csv"),
    ]

    # ---- train -------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"], callbacks=callbacks, verbose=0)
    train_seconds = time.perf_counter() - t0
    peak_mb = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1024**2 if gpus else None

    hist = history.history
    seconds = hist["seconds"]
    best_epoch = int(np.argmax(hist[MONITOR])) + 1
    training = {
        "epochs_trained": len(seconds),
        "max_epochs": cfg["epochs"],
        "stopped_early": len(seconds) < cfg["epochs"],
        "best_epoch": best_epoch,
        "best_val_water_iou": float(np.max(hist[MONITOR])),
        "final_learning_rate": float(hist["learning_rate"][-1]),
        # epoch 1 includes graph building and cuDNN autotuning, so it is reported separately
        "sec_per_epoch": round(float(np.median(seconds[1:] if len(seconds) > 1 else seconds)), 2),
        "sec_first_epoch": round(float(seconds[0]), 2),
        "train_minutes": round(train_seconds / 60, 2),
        "peak_gpu_memory_mb": round(peak_mb, 1) if peak_mb is not None else None,
        "parameters": params["total_params"],
        "trainable_parameters": params["trainable_params"],
    }

    # ---- keep the best-validation model in an inference-ready file ----------------------------------
    best = build_unet(cfg["encoder"], n_channels(cfg["channels"]), bn_momentum=cfg["bn_momentum"])
    best.load_weights(weights)
    best.save(out / "best.keras")
    weights.unlink()

    # ---- exact validation scores of that model (tested NumPy metrics, full chips) --------------------
    val, table = evaluate_split(best, "val", cfg)
    gap = abs(val["iou"] - training["best_val_water_iou"])
    if gap > 2e-3:  # the Keras metric and the NumPy metric must agree; if not, something is broken
        raise RuntimeError(f"{run_id}: validation IoU of the reloaded model ({val['iou']:.4f}) differs from "
                           f"the training log ({training['best_val_water_iou']:.4f})")
    table.to_csv(out / "per_chip.csv", index=False)
    metrics = {"run_id": run_id, "training": training, "val": val}
    write_json(out / "metrics.json", metrics)
    log(summary_line(run_id, metrics))

    del model, best, train_ds, val_ds, history
    keras.backend.clear_session()
    gc.collect()
    return json.loads((out / "metrics.json").read_text())


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main():
    p = argparse.ArgumentParser(description="Train one FloodWatch run.")
    p.add_argument("--config", required=True, help="run id (R04) or path to a run config (configs/R04.json)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config values for an ad-hoc run, e.g. --set epochs=5 seed=3")
    p.add_argument("--out", help="output folder (default: results/<run_id>)")
    p.add_argument("--force", action="store_true", help="retrain even if the run folder is complete")
    p.add_argument("--evaluate", action="store_true", help="also score test and Bolivia when training is done")
    args = p.parse_args()

    cfg = load_config(args.config)
    overrides = dict(item.split("=", 1) for item in args.set)
    if overrides:
        cfg = resolve({**cfg, **{k: parse_value(v) for k, v in overrides.items()}})
    out = ROOT / args.out if args.out else None
    train_run(cfg, out_dir=out, force=args.force)
    if args.evaluate:
        from src.evaluate import evaluate_run

        evaluate_run(cfg["run_id"])


if __name__ == "__main__":
    main()
