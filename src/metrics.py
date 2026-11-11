"""Segmentation metrics for water masks, computed over valid pixels only.

Label convention (see src/data.py): -1 = no data, 0 = non-water, 1 = water.
`pred` is a binary mask (1 = predicted water), same shape as `label`.

Two IoU numbers are reported everywhere, and they are NOT the same thing:

    water IoU (pooled)   TP, FP, FN are summed over every chip of the split, then IoU = TP / (TP + FP + FN)
                         once. Large flooded chips dominate. This is the primary metric.
    mean per-chip IoU    IoU of each chip, then the plain average. Every chip weighs the same.

Rule for chips with no water and no predicted water
---------------------------------------------------
IoU is 0/0 for a chip where, among valid pixels, the label has no water AND the prediction has no
water (TP + FP + FN == 0). Such a chip is SKIPPED in the mean per-chip IoU (it is not counted as 0 or
as 1). A chip that has water in the label but none predicted (or the reverse) is NOT skipped: its IoU
is 0 and it is averaged in. The same applies to a chip with no valid pixels at all. If every chip is
skipped the mean is NaN.
"""
import numpy as np
import pandas as pd

NO_DATA = -1
WATER = 1


def confusion_counts(pred, label):
    """Return (tp, fp, fn) as ints, counting only pixels where label != -1."""
    pred = np.asarray(pred)
    label = np.asarray(label)
    if pred.shape != label.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs label {label.shape}")
    valid = label != NO_DATA
    p = (pred == WATER) & valid
    t = (label == WATER) & valid
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t & valid).sum())
    return tp, fp, fn


def _div(num, den):
    return num / den if den > 0 else float("nan")


def scores_from_counts(tp, fp, fn):
    """IoU, precision, recall, F1 from counts. A 0/0 term gives NaN."""
    iou = _div(tp, tp + fp + fn)
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    # 2PR/(P+R) written in counts: 2TP / (2TP + FP + FN). Same value, no NaN-propagation
    # when P or R is 0/0 but the other is defined.
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def chip_scores(pred, label):
    """Metrics for a single chip (also returns the raw counts)."""
    tp, fp, fn = confusion_counts(pred, label)
    return {"tp": tp, "fp": fp, "fn": fn, **scores_from_counts(tp, fp, fn)}


def pooled_scores(preds, labels):
    """Metrics from TP/FP/FN summed over all chips (one global confusion matrix)."""
    tp = fp = fn = 0
    for p, l in zip(preds, labels):
        a, b, c = confusion_counts(p, l)
        tp, fp, fn = tp + a, fp + b, fn + c
    return {"tp": tp, "fp": fp, "fn": fn, **scores_from_counts(tp, fp, fn)}


def mean_chip_iou(preds, labels):
    """Average of per-chip IoU, skipping chips with TP + FP + FN == 0 (see module docstring).

    Returns (mean_iou, n_used, n_skipped).
    """
    ious = []
    skipped = 0
    for p, l in zip(preds, labels):
        tp, fp, fn = confusion_counts(p, l)
        if tp + fp + fn == 0:
            skipped += 1
            continue
        ious.append(tp / (tp + fp + fn))
    mean = float(np.mean(ious)) if ious else float("nan")
    return mean, len(ious), skipped


# --------------------------------------------------------------------------------------------------
# Table form: one row per chip. Every reported number can be rebuilt from these rows.
# --------------------------------------------------------------------------------------------------

def per_chip_table(preds, labels, names=None):
    """One row per chip: valid / water pixel counts, TP / FP / FN and the four scores."""
    rows = []
    for i, (p, l) in enumerate(zip(preds, labels)):
        l = np.asarray(l)
        n_valid = int((l != NO_DATA).sum())
        n_water = int((l == WATER).sum())
        rows.append({
            "chip": names[i] if names is not None else i,
            "n_valid": n_valid,
            "n_water": n_water,
            "water_frac": n_water / n_valid if n_valid else float("nan"),
            **chip_scores(p, l),
        })
    return pd.DataFrame(rows)


def summarise(table):
    """Split-level scores from a per-chip table (needs columns tp, fp, fn).

    Returns pooled IoU / precision / recall / F1 plus the mean per-chip IoU and how many chips it used.
    """
    tp, fp, fn = (int(table[c].sum()) for c in ("tp", "fp", "fn"))
    defined = (table["tp"] + table["fp"] + table["fn"]) > 0
    per_chip = table.loc[defined, "tp"] / (table.loc[defined, "tp"] + table.loc[defined, "fp"] + table.loc[defined, "fn"])
    return {
        "n_chips": int(len(table)),
        **scores_from_counts(tp, fp, fn),
        "mean_chip_iou": float(per_chip.mean()) if len(per_chip) else float("nan"),
        "chips_in_mean": int(defined.sum()),
        "chips_skipped": int((~defined).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def bootstrap_iou_ci(table, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap interval for the pooled water IoU, resampling CHIPS with replacement.

    This measures how much the score depends on which chips happen to be in the split (90 test chips,
    15 Bolivia chips); it is a different source of uncertainty from the seed-to-seed std.
    Returns (low, high).
    """
    counts = table[["tp", "fp", "fn"]].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    sums = counts[idx].sum(axis=1)  # (n_boot, 3)
    den = sums.sum(axis=1)
    iou = np.divide(sums[:, 0], den, out=np.full(n_boot, np.nan), where=den > 0)
    low, high = np.nanquantile(iou, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
