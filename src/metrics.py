"""Segmentation metrics for water masks, computed over valid pixels only.

Label convention (see src/data.py): -1 = no data, 0 = non-water, 1 = water.
`pred` is a binary mask (1 = predicted water), same shape as `label`.

Rule for chips with no water and no predicted water
---------------------------------------------------
IoU is 0/0 for a chip where, among valid pixels, the label has no water AND the
prediction has no water (TP + FP + FN == 0). Such a chip is SKIPPED in the mean
per-chip IoU (it is not counted as 0 or as 1). A chip that has water in the label
but none predicted (or the reverse) is NOT skipped: its IoU is 0 and it is averaged
in. The same applies to a chip with no valid pixels at all. If every chip is skipped
the mean is NaN.
"""
import numpy as np

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
