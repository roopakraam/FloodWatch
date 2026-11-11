"""R00: Otsu threshold baseline on VH.

Threshold choice: ONE global threshold, computed from the VH pixels of all training chips.
Why not one per chip: many chips have little or no water (32 of 252 train chips have none),
and Otsu on a histogram with a single peak still forces a split, so it would cut through the
land and label a big chunk of it as water. A global threshold is also a single fixed rule we
can apply to unseen chips, which is what the later models have to do too.

Prediction: water where VH < threshold (NaN VH pixels are never water).

Run from the project root:  python -m src.otsu_baseline
"""
import json

import numpy as np
from skimage.filters import threshold_otsu

from src.data import ROOT, list_split, load_chip
from src.metrics import mean_chip_iou, pooled_scores

BINS = np.linspace(-35, 5, 401)  # 0.1 dB bins over the clip range used elsewhere
OUT_JSON = ROOT / "results" / "otsu_baseline.json"


def train_threshold():
    """Otsu threshold on the pooled VH histogram of the train split."""
    counts = np.zeros(len(BINS) - 1, dtype=np.int64)
    for s1_path, label_path in list_split("train"):
        image, _ = load_chip(s1_path, label_path, nan_value=None)
        vh = image[1]
        vh = np.clip(vh[~np.isnan(vh)], BINS[0], BINS[-1])
        counts += np.histogram(vh, bins=BINS)[0]
    centers = (BINS[:-1] + BINS[1:]) / 2
    return float(threshold_otsu(hist=(counts, centers)))


def evaluate(split, threshold):
    preds, labels = [], []
    for s1_path, label_path in list_split(split):
        image, label = load_chip(s1_path, label_path, nan_value=None)
        preds.append((image[1] < threshold).astype(np.int8))  # NaN < t is False
        labels.append(label)
    pooled = pooled_scores(preds, labels)
    miou, n_used, n_skipped = mean_chip_iou(preds, labels)
    return {
        "n_chips": len(preds),
        "iou": pooled["iou"],
        "precision": pooled["precision"],
        "recall": pooled["recall"],
        "f1": pooled["f1"],
        "mean_chip_iou": miou,
        "chips_in_mean": n_used,
        "chips_skipped": n_skipped,
    }


def main():
    thr = train_threshold()
    print(f"Otsu threshold on train VH: {thr:.2f} dB")
    out = {"run": "R00", "model": "Otsu on VH", "threshold_db": thr, "threshold_source": "global, train VH pixels"}
    for split in ("test", "bolivia"):
        out[split] = evaluate(split, thr)
        r = out[split]
        print(f"{split:8s} IoU {r['iou']:.4f}  P {r['precision']:.4f}  R {r['recall']:.4f}  "
              f"F1 {r['f1']:.4f}  mean-chip IoU {r['mean_chip_iou']:.4f} "
              f"({r['chips_in_mean']} chips, {r['chips_skipped']} skipped)")
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
