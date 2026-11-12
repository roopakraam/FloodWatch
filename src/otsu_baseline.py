"""R00 (experiment E0): Otsu threshold baseline on VH.

Rule: water where VH < threshold. Pixels with no VH value (NaN) are never water.

Threshold choice: ONE global threshold, computed with Otsu's method from the VH values of all
training chips (no labels involved). Why not one threshold per chip: Otsu always splits a histogram
in two, also when a chip holds no water at all (32 of the 252 training chips), where it cuts straight
through the land and calls the darker half water. A single fixed threshold is also what a deployed
rule would be: decided once, then applied to images it has never seen. The numbers behind this
choice are in notebooks/02_otsu_baseline.ipynb.

The baseline is scored with exactly the same code as every U-Net (src.evaluate.score_masks) and
writes the same files: results/R00/{config.json, metrics.json, per_chip.csv}.

R00b, a supplementary reference that was not in the original plan: the single VH threshold that gives
the highest pooled IoU on the TRAINING labels. It is the strongest rule of the form "VH < t" that can be
chosen without looking at validation or test data, so it shows how far thresholding itself can go.

Run from the project root:  python -m src.otsu_baseline
"""
import json
from datetime import datetime, timezone

import numpy as np
from skimage.filters import threshold_otsu

from src.config import run_dir
from src.data import WATER, load_split
from src.evaluate import score_masks, write_json
from src.preprocess import CLIP_DB

RUN_ID = "R00"
TUNED_RUN_ID = "R00b"
EDGES = np.linspace(CLIP_DB[0], CLIP_DB[1], 401)  # 0.1 dB bins over the clip range used everywhere
SWEEP = np.round(np.arange(-30.0, -10.0 + 1e-9, 0.05), 2)  # candidate thresholds for the tuned reference
SCORED_SPLITS = ("val", "test", "bolivia")


def vh_histogram(split="train"):
    """Histogram of all finite VH values of a split, clipped to CLIP_DB. Returns (counts, bin centers)."""
    vh = load_split(split).images[:, 1]
    counts = np.histogram(np.clip(vh[np.isfinite(vh)], *CLIP_DB), bins=EDGES)[0]
    return counts, (EDGES[:-1] + EDGES[1:]) / 2


def train_threshold():
    """Otsu threshold (dB) on the pooled VH histogram of the training split."""
    counts, centers = vh_histogram("train")
    return float(threshold_otsu(hist=(counts, centers)))


def predict(images, threshold):
    """(N, 2, H, W) raw dB images -> (N, H, W) int8 water masks. NaN < threshold is False."""
    with np.errstate(invalid="ignore"):
        return (images[:, 1] < threshold).astype(np.int8)


def evaluate(split, threshold):
    """Score the threshold rule on one split. Returns (summary dict, per-chip DataFrame)."""
    data = load_split(split)
    return score_masks(predict(data.images, threshold), data, split)


def threshold_sweep(split, thresholds):
    """Pooled water IoU of the rule 'VH < t' for every t in `thresholds` (a diagnostic, not a model).

    Returns an array of IoU values. Used on train / validation only, to see how far ANY single VH
    threshold can go.
    """
    data = load_split(split)
    vh, label = data.images[:, 1], data.labels
    water, land = vh[label == WATER], vh[label == 0]
    n_water = water.size                       # includes water pixels with no VH value: always missed
    water, land = np.sort(water[np.isfinite(water)]), np.sort(land[np.isfinite(land)])
    tp = np.searchsorted(water, thresholds, side="left")
    fp = np.searchsorted(land, thresholds, side="left")
    return tp / (n_water + fp)                 # TP / (TP + FN + FP) with TP + FN = all water pixels


def tuned_threshold():
    """The threshold in SWEEP with the highest pooled water IoU on the training labels."""
    return float(SWEEP[int(np.argmax(threshold_sweep("train", SWEEP)))])


def per_chip_otsu_false_alarm(split="train"):
    """What a PER-CHIP Otsu threshold does on chips that contain no water: share of labelled pixels it
    would call water on each such chip (all of them false alarms)."""
    data = load_split(split)
    shares = []
    for image, label in zip(data.images, data.labels):
        if (label == WATER).any() or not (label == 0).any():
            continue
        vh = image[1]
        finite = np.isfinite(vh)
        t = threshold_otsu(np.clip(vh[finite], *CLIP_DB))
        shares.append(float(((vh < t) & finite & (label == 0)).sum() / (label == 0).sum()))
    return np.array(shares)


def score_threshold_rule(run_id, config_name, study, threshold, method, source, log=print):
    """Score the rule 'VH < threshold' on validation, test and Bolivia and write results/<run_id>/."""
    import pandas as pd

    out = run_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {
        "run_id": run_id, "config_name": config_name, "study": study, "method": method,
        "rule": "water where VH < threshold; NaN is never water", "threshold_db": threshold,
        "threshold_source": source, "channels": "vh", "trainable_parameters": 0,
    })
    metrics, tables = {"run_id": run_id, "threshold_db": threshold}, []
    for split in SCORED_SPLITS:
        summary, table = evaluate(split, threshold)
        metrics[split] = summary
        tables.append(table)
        log(f"{run_id} {split:8s} IoU {summary['iou']:.4f}  P {summary['precision']:.4f}  R {summary['recall']:.4f}  "
            f"F1 {summary['f1']:.4f}  mean per-chip IoU {summary['mean_chip_iou']:.4f} "
            f"({summary['chips_in_mean']} chips, {summary['chips_skipped']} skipped)")
    metrics["evaluated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(out / "metrics.json", metrics)
    pd.concat(tables, ignore_index=True).to_csv(out / "per_chip.csv", index=False)
    return json.loads((out / "metrics.json").read_text())


def main():
    """R00 (Otsu, the planned baseline) and R00b (threshold tuned on the training labels)."""
    otsu = train_threshold()
    print(f"Otsu threshold on the training VH pixels: {otsu:.2f} dB")
    r00 = score_threshold_rule(
        RUN_ID, "Otsu (E0)", "E0", otsu, "global Otsu threshold on VH",
        "Otsu on the pooled VH histogram of the 252 training chips (0.1 dB bins, clipped to [-35, 5] dB); no labels used")
    tuned = tuned_threshold()
    print(f"\nThreshold with the best pooled IoU on the training labels: {tuned:.2f} dB")
    r00b = score_threshold_rule(
        TUNED_RUN_ID, "Tuned VH threshold", "E0 (extra)", tuned, "global VH threshold tuned on the training labels",
        "threshold between -30 and -10 dB (0.05 dB steps) with the highest pooled water IoU on the 252 training chips")
    return r00, r00b


if __name__ == "__main__":
    main()
