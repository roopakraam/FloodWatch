"""Score a trained run on full 512x512 chips (threshold 0.5) with the tested NumPy metrics.

Validation is scored right after training (src/train.py). Test and Bolivia are scored ONCE per finished
run by `evaluate_run`, which refuses to overwrite an existing test score unless forced: re-evaluating
until the number looks good would be tuning on the test set.

Everything a later table or figure needs is written next to the model:

    results/Rxx/metrics.json   split-level scores (pooled water IoU / precision / recall / F1, mean per-chip IoU)
    results/Rxx/per_chip.csv   one row per chip and split: pixel counts, TP / FP / FN, per-chip scores

`observed_only` repeats the scores after dropping pixels whose SAR value is NaN. Those pixels have a
label but no radar measurement (one test chip, Paraguay_34417, has no radar data at all), so no method
can get them right from the image; the headline numbers keep them to stay comparable with other work.

Run from the project root:
    python -m src.evaluate --run R04
"""
if __package__ in (None, ""):  # allow `python src/evaluate.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import load_config, run_dir
from src.data import NO_DATA, event_of, load_split
from src.metrics import per_chip_table, summarise
from src.preprocess import finite_mask
from src.runlog import score_line

TEST_SPLITS = ("test", "bolivia")
SCORE_KEYS = ("iou", "precision", "recall", "f1")


def to_jsonable(obj):
    """NaN -> None and NumPy scalars -> Python, so the files are strict JSON."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_json(path, obj):
    path.write_text(json.dumps(to_jsonable(obj), indent=2) + "\n")


def read_metrics(run_id):
    path = run_dir(run_id) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_run_model(run_id):
    """The best-validation model of a run, ready for inference (no optimizer, no custom objects needed)."""
    import keras

    return keras.models.load_model(run_dir(run_id) / "best.keras", compile=False)


def predict_proba(model, X, batch=8):
    """Water probability for every pixel: X (N, H, W, C) -> (N, H, W) float32.

    The last batch is padded to the full batch size (and the padding thrown away) so the XLA-compiled
    predict function sees one input shape only. In inference mode every chip is processed independently,
    so padding cannot change the result.
    """
    out = np.empty(X.shape[:3], dtype=np.float32)
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        n = len(xb)
        if n < batch:
            xb = np.concatenate([xb, np.repeat(xb[-1:], batch - n, axis=0)])
        out[i:i + n] = np.asarray(model.predict_on_batch(xb), dtype=np.float32)[:n, ..., 0]
    return out


def score_masks(pred, data, split):
    """Binary masks (N, H, W) + the raw split -> (summary dict, per-chip DataFrame)."""
    table = per_chip_table(pred, data.labels, data.names)
    table.insert(0, "split", split)
    table.insert(2, "event", [event_of(n) for n in data.names])

    # Sensitivity: the same scores without the pixels that have a label but no SAR measurement.
    observed = finite_mask(data.images)
    table_obs = per_chip_table(pred, np.where(observed, data.labels, NO_DATA), data.names)
    table["n_valid_no_sar"] = table["n_valid"] - table_obs["n_valid"]
    for c in ("tp", "fp", "fn"):
        table[f"{c}_obs"] = table_obs[c]

    summary = summarise(table)
    obs = summarise(table_obs)
    summary["observed_only"] = {k: obs[k] for k in (*SCORE_KEYS, "mean_chip_iou")}
    summary["n_valid_px"] = int(table["n_valid"].sum())
    summary["n_valid_px_no_sar"] = int(table["n_valid_no_sar"].sum())
    return summary, table


def evaluate_split(model, split, cfg, time_it=False):
    """Predict one split with `model` and score it. Returns (summary, per-chip table)."""
    from src.pipeline import load_arrays

    X, _ = load_arrays(split, cfg["channels"])
    batch = cfg["eval_batch_size"]
    if time_it:  # warm-up so graph building is not counted
        model.predict_on_batch(X[:batch])
    t0 = time.perf_counter()
    probs = predict_proba(model, X, batch)
    seconds = time.perf_counter() - t0
    pred = (probs >= cfg["threshold"]).astype(np.int8)
    summary, table = score_masks(pred, load_split(split), split)
    if time_it:
        summary["inference_ms_per_chip"] = round(1000 * seconds / len(X), 2)
    return summary, table


def save_per_chip(run_id, table):
    """Add / replace the rows of the split(s) in `table` inside results/Rxx/per_chip.csv."""
    path = run_dir(run_id) / "per_chip.csv"
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[~old["split"].isin(table["split"].unique())], table], ignore_index=True)
    table.to_csv(path, index=False)


def evaluate_run(run_id, splits=TEST_SPLITS, force=False, log=print):
    """Score a finished run on `splits` (default: test and Bolivia), once. Returns the metrics dict."""
    metrics = read_metrics(run_id)
    if "training" not in metrics:
        raise RuntimeError(f"{run_id} has not been trained yet (no training summary in metrics.json)")
    todo = [s for s in splits if force or s not in metrics]
    if not todo:
        log(f"[{run_id} was already scored on {', '.join(splits)}; not scoring it again]")
        return metrics

    cfg = json.loads((run_dir(run_id) / "config.json").read_text())
    model = load_run_model(run_id)
    for split in todo:
        summary, table = evaluate_split(model, split, cfg, time_it=True)
        metrics[split] = summary
        save_per_chip(run_id, table)
        log(score_line(run_id, split, summary))
    metrics["evaluated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(run_dir(run_id) / "metrics.json", metrics)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True, help="run id, e.g. R04")
    p.add_argument("--force", action="store_true", help="re-evaluate even if test scores exist (say why in the notes!)")
    args = p.parse_args()

    from src.train import setup_runtime

    setup_runtime(load_config(args.run)["mixed_precision"])
    evaluate_run(args.run, force=args.force)


if __name__ == "__main__":
    main()
