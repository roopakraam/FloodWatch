"""Collect the per-run result files into tables.

The source of truth for every reported number is results/<run_id>/{config.json, metrics.json,
per_chip.csv}. This module only reads those files:

    runs_table()        one row per run, laid out like the "Runs" tab of the project tracker
                        (same column names, in the same order) plus a few extra columns
    summary_table()     one row per config: mean and standard deviation over its finished seeds
    study_table(study)  the configs of one ablation study next to each other
    per_chip(run_ids)   the per-chip rows of several runs stacked in one frame

Run from the project root to (re)write results/all_runs.csv and results/summary.csv:
    python -m src.analysis
"""
import json

import numpy as np
import pandas as pd

from src.config import (CHANNEL_LABELS, CONFIG_NAMES, LOSS_LABELS, RESULTS_DIR, STUDIES, default_config, run_dir,
                        run_table)

BASELINE = {"run_id": "R00", "config_name": "Otsu (E0)", "study": "E0", "seed": None}
# Supplementary reference, not one of the 22 planned runs: the best single VH threshold on the training labels.
TUNED_BASELINE = {"run_id": "R00b", "config_name": "Tuned VH threshold", "study": "E0 (extra)", "seed": None}
THRESHOLD_RUNS = ("R00", "R00b")
ALL_CONFIGS = ("Otsu (E0)", "Tuned VH threshold", *CONFIG_NAMES)

# The first 20 columns are exactly the header of the tracker's "Runs" tab.
TRACKER_COLUMNS = ["Run ID", "Config", "Seed", "Status", "Test IoU", "Bolivia IoU", "Val IoU (best)", "Test F1",
                   "Test precision", "Test recall", "Bolivia F1", "Study", "Channels", "Loss", "Encoder",
                   "Epochs trained", "Sec per epoch", "Peak GPU memory (MB)", "Parameters", "Notes"]
EXTRA_COLUMNS = ["Best epoch", "Train minutes", "Peak host RAM (MB)", "Inference ms per chip",
                 "Val mean per-chip IoU", "Test mean per-chip IoU", "Bolivia mean per-chip IoU",
                 "Bolivia precision", "Bolivia recall", "Test IoU (observed pixels only)",
                 "Bolivia IoU (observed pixels only)", "Git commit"]
METRIC_COLUMNS = ["Val IoU (best)", "Test IoU", "Test precision", "Test recall", "Test F1", "Test mean per-chip IoU",
                  "Bolivia IoU", "Bolivia precision", "Bolivia recall", "Bolivia F1", "Bolivia mean per-chip IoU"]
ENCODER_SHORT = {"scratch": "Scratch (16–256 filters)", "mobilenetv2": "MobileNetV2",
                 "mobilenetv2_imagenet": "MobileNetV2 + ImageNet weights"}


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def run_row(run):
    """One tracker-style row for a planned run (a dict from config.run_table() or BASELINE)."""
    run_id = run["run_id"]
    metrics = _read(run_dir(run_id) / "metrics.json") or {}
    cfg = _read(run_dir(run_id) / "config.json") or {**default_config(), **run}
    training = metrics.get("training", {})
    is_baseline = run_id in THRESHOLD_RUNS
    finished = "test" in metrics and "bolivia" in metrics
    status = "Finished" if finished else ("Running" if (training or (run_dir(run_id) / "history.csv").exists()) else "Not run")

    def score(split, key):
        return metrics.get(split, {}).get(key, np.nan)

    if run_id == BASELINE["run_id"]:
        notes = (f"No training. One global Otsu threshold from all training VH pixels = "
                 f"{metrics.get('threshold_db', float('nan')):.2f} dB; water where VH < threshold.")
    elif run_id == TUNED_BASELINE["run_id"]:
        notes = (f"Extra reference, not in the original plan. No training. The single VH threshold with the best "
                 f"pooled IoU on the training labels = {metrics.get('threshold_db', float('nan')):.2f} dB.")
    elif training:
        how = (f"early stop at epoch {training['epochs_trained']}" if training["stopped_early"]
               else f"ran all {training['epochs_trained']} epochs")
        notes = f"{how}, best epoch {training['best_epoch']}."
    else:
        notes = ""

    return {
        "Run ID": run_id,
        "Config": run["config_name"],
        "Seed": "–" if is_baseline else run["seed"],
        "Status": status,
        "Test IoU": score("test", "iou"),
        "Bolivia IoU": score("bolivia", "iou"),
        "Val IoU (best)": score("val", "iou"),
        "Test F1": score("test", "f1"),
        "Test precision": score("test", "precision"),
        "Test recall": score("test", "recall"),
        "Bolivia F1": score("bolivia", "f1"),
        "Study": run["study"],
        "Channels": "VH" if is_baseline else CHANNEL_LABELS[cfg["channels"]],
        "Loss": "none (threshold)" if is_baseline else LOSS_LABELS[cfg["loss"]],
        "Encoder": "none" if is_baseline else ENCODER_SHORT[cfg["encoder"]],
        "Epochs trained": training.get("epochs_trained", np.nan),
        "Sec per epoch": training.get("sec_per_epoch", np.nan),
        "Peak GPU memory (MB)": training.get("peak_gpu_memory_mb", np.nan),
        "Parameters": 0 if is_baseline else training.get("parameters", np.nan),
        "Notes": notes,
        "Best epoch": training.get("best_epoch", np.nan),
        "Train minutes": training.get("train_minutes", np.nan),
        "Peak host RAM (MB)": training.get("peak_host_ram_mb", np.nan),
        "Inference ms per chip": score("test", "inference_ms_per_chip"),
        "Val mean per-chip IoU": score("val", "mean_chip_iou"),
        "Test mean per-chip IoU": score("test", "mean_chip_iou"),
        "Bolivia mean per-chip IoU": score("bolivia", "mean_chip_iou"),
        "Bolivia precision": score("bolivia", "precision"),
        "Bolivia recall": score("bolivia", "recall"),
        "Test IoU (observed pixels only)": metrics.get("test", {}).get("observed_only", {}).get("iou", np.nan),
        "Bolivia IoU (observed pixels only)": metrics.get("bolivia", {}).get("observed_only", {}).get("iou", np.nan),
        "Git commit": (cfg.get("provenance") or {}).get("git_commit"),
    }


def runs_table(finished_only=False):
    """R00 ... R21 (and the extra reference R00b last), one row each, in the tracker's column layout."""
    table = pd.DataFrame([run_row(r) for r in (BASELINE, *run_table(), TUNED_BASELINE)])[TRACKER_COLUMNS + EXTRA_COLUMNS]
    return table[table["Status"] == "Finished"].reset_index(drop=True) if finished_only else table


def summary_table(runs=None):
    """Mean and sample standard deviation (ddof = 1) over the finished seeds of every config."""
    runs = runs_table() if runs is None else runs
    done = runs[runs["Status"] == "Finished"]
    rows = []
    for config in ALL_CONFIGS:
        r = done[done["Config"] == config]
        row = {"Config": config, "Finished runs": len(r)}
        for col in METRIC_COLUMNS + ["Epochs trained", "Sec per epoch", "Peak GPU memory (MB)", "Parameters",
                                     "Inference ms per chip"]:
            values = pd.to_numeric(r[col], errors="coerce").dropna()
            row[f"{col} mean"] = values.mean() if len(values) else np.nan
            row[f"{col} std"] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std(mean, std, digits=3):
    """'0.712 ± 0.008' (or just the mean when there is a single run)."""
    if pd.isna(mean):
        return ""
    return f"{mean:.{digits}f}" + ("" if pd.isna(std) else f" ± {std:.{digits}f}")


def study_table(study, summary=None, metrics=("Val IoU (best)", "Test IoU", "Test precision", "Test recall",
                                              "Bolivia IoU")):
    """The configs of one study side by side as 'mean ± std' strings, with the Otsu baseline on top."""
    summary = summary_table() if summary is None else summary
    order = ["Otsu (E0)", "Tuned VH threshold", *STUDIES[study]["configs"]]
    s = summary.set_index("Config").loc[order]
    out = pd.DataFrame({"Finished runs": s["Finished runs"]})
    for m in metrics:
        out[m] = [mean_std(a, b) for a, b in zip(s[f"{m} mean"], s[f"{m} std"])]
    return out


def gap_table(study, runs=None, metrics=("Test IoU", "Bolivia IoU")):
    """Every pair of configs in a study: difference of the seed means next to the seed-to-seed std.

    The project's rule: a gap that is smaller than the standard deviation over seeds is not a real
    difference. `Larger std` is the larger of the two configs' stds; Welch's t-test is added for
    reference only (three seeds per config give it very little power).
    """
    from itertools import combinations

    from scipy import stats

    runs = runs_table() if runs is None else runs
    done = runs[runs["Status"] == "Finished"]
    rows = []
    for a, b in combinations(STUDIES[study]["configs"], 2):
        for metric in metrics:
            va = pd.to_numeric(done.loc[done["Config"] == a, metric]).to_numpy()
            vb = pd.to_numeric(done.loc[done["Config"] == b, metric]).to_numpy()
            if len(va) < 2 or len(vb) < 2:
                continue
            gap, noise = va.mean() - vb.mean(), max(va.std(ddof=1), vb.std(ddof=1))
            rows.append({"Metric": metric, "A": a, "B": b, "Mean A": va.mean(), "Mean B": vb.mean(), "Gap (A - B)": gap,
                         "Larger std": noise, "Gap / std": abs(gap) / noise if noise > 0 else np.inf,
                         "Gap larger than std": bool(abs(gap) > noise),
                         "Welch p": float(stats.ttest_ind(va, vb, equal_var=False).pvalue)})
    return pd.DataFrame(rows)


def histories(run_ids):
    """{run_id: history DataFrame} for the runs that have one."""
    out = {}
    for run_id in run_ids:
        path = run_dir(run_id) / "history.csv"
        if path.exists():
            out[run_id] = pd.read_csv(path)
    return out


def per_chip(run_ids, split=None):
    """Stack results/<run>/per_chip.csv of several runs (adds a run_id column)."""
    frames = []
    for run_id in run_ids:
        path = run_dir(run_id) / "per_chip.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "run_id", run_id)
            frames.append(frame if split is None else frame[frame["split"] == split])
    return pd.concat(frames, ignore_index=True)


def best_run_by_validation(runs=None, config=None):
    """The finished U-Net run with the highest VALIDATION IoU (never chosen by test)."""
    runs = runs_table() if runs is None else runs
    r = runs[(runs["Status"] == "Finished") & ~runs["Run ID"].isin(THRESHOLD_RUNS)]
    if config is not None:
        r = r[r["Config"] == config]
    return r.loc[r["Val IoU (best)"].idxmax(), "Run ID"]


def write_tables():
    runs = runs_table()
    runs.to_csv(RESULTS_DIR / "all_runs.csv", index=False)
    summary = summary_table(runs)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return runs, summary


if __name__ == "__main__":
    runs, summary = write_tables()
    print(runs[["Run ID", "Config", "Seed", "Status", "Val IoU (best)", "Test IoU", "Bolivia IoU"]].to_string(index=False))
    print(f"\nwrote {RESULTS_DIR / 'all_runs.csv'} and {RESULTS_DIR / 'summary.csv'}")
