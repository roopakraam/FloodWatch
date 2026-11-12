"""The per-epoch log lines of a run: formatting, and replaying a finished run from its saved files.

Kept free of TensorFlow so that the notebooks can show the log of a finished run instantly.
"""
import json
from pathlib import Path

import pandas as pd


def is_trained(out_dir):
    path = Path(out_dir) / "metrics.json"
    return path.exists() and "training" in json.loads(path.read_text())


def header_line(cfg, n_params):
    policy = "mixed_float16" if cfg["mixed_precision"] else "float32"
    return (f"{cfg.get('run_id') or 'adhoc'}: {cfg.get('config_name') or ''} | encoder {cfg['encoder']} | "
            f"channels {cfg['channels']} | loss {cfg['loss']} | seed {cfg['seed']} | {n_params:,} parameters | {policy}")


def format_epoch(row, total_epochs):
    """One log line from a history row (a dict or a pandas Series)."""
    star = " *" if row.get("is_best") else ""
    return (f"epoch {int(row['epoch']) + 1:2d}/{total_epochs} | loss {row['loss']:.4f}  iou {row['water_iou']:.4f} | "
            f"val_loss {row['val_loss']:.4f}  val_iou {row['val_water_iou']:.4f} | "
            f"lr {row['learning_rate']:.1e} | {row['seconds']:5.1f}s{star}")


def summary_line(run_id, metrics):
    t, v = metrics["training"], metrics["val"]
    return (f"{run_id}: stopped after {t['epochs_trained']} epochs, best epoch {t['best_epoch']}, "
            f"val IoU {v['iou']:.4f} (mean per-chip {v['mean_chip_iou']:.4f}) | "
            f"{t['sec_per_epoch']:.1f} s/epoch, {t['train_minutes']:.1f} min, "
            f"peak GPU memory {t['peak_gpu_memory_mb']} MB")


def score_line(run_id, split, s):
    return (f"{run_id} {split:8s} IoU {s['iou']:.4f}  P {s['precision']:.4f}  R {s['recall']:.4f}  F1 {s['f1']:.4f}  "
            f"mean per-chip IoU {s['mean_chip_iou']:.4f} ({s['chips_in_mean']} chips)")


def replay(out_dir, log=print, scores=("test", "bolivia")):
    """Print the saved log of a finished run: header, one line per epoch, summary, and its test scores."""
    out = Path(out_dir)
    cfg = json.loads((out / "config.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    run_id = cfg.get("run_id") or "adhoc"
    started = cfg["provenance"]["started_utc"][:16].replace("T", " ")
    log(f"[{run_id} was trained on {started} UTC; this is its saved log]")
    log(header_line(cfg, metrics["training"]["parameters"]))
    for _, row in pd.read_csv(out / "history.csv").iterrows():
        log(format_epoch(row, cfg["epochs"]))
    log(summary_line(run_id, metrics))
    for split in scores:
        if split in metrics:
            log(score_line(run_id, split, metrics[split]))
    return metrics
