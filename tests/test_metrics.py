"""Hand-checked 3x3 example for src/metrics.py. Run: python -m pytest tests  (or python -m tests.test_metrics)

label (-1 = no data)      pred
 1  1  0                  1  0  1
 0 -1  1                  0  1  1
 0  0 -1                  1  0  0

Pixel by pixel (row, col):
 (0,0) l1 p1 -> TP        (0,1) l1 p0 -> FN        (0,2) l0 p1 -> FP
 (1,0) l0 p0 -> TN        (1,1) NO DATA, ignored   (1,2) l1 p1 -> TP
 (2,0) l0 p1 -> FP        (2,1) l0 p0 -> TN        (2,2) NO DATA, ignored
Note (1,1) is predicted water but must NOT count as FP.

TP=2, FP=2, FN=1
IoU = 2/5 = 0.4 ; precision = 2/4 = 0.5 ; recall = 2/3
F1 = 2*0.5*(2/3) / (0.5 + 2/3) = (2/3)/(7/6) = 4/7
"""
import math

import numpy as np

from src.metrics import (bootstrap_iou_ci, chip_scores, confusion_counts, mean_chip_iou, per_chip_table,
                         pooled_scores, summarise)

LABEL = np.array([[1, 1, 0], [0, -1, 1], [0, 0, -1]])
PRED = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])


def test_counts_ignore_no_data():
    assert confusion_counts(PRED, LABEL) == (2, 2, 1)


def test_scores_match_paper():
    s = chip_scores(PRED, LABEL)
    assert math.isclose(s["iou"], 2 / 5)
    assert math.isclose(s["precision"], 1 / 2)
    assert math.isclose(s["recall"], 2 / 3)
    assert math.isclose(s["f1"], 4 / 7)


def test_mean_chip_iou_skips_empty_chips():
    empty_label = np.zeros((3, 3), dtype=int)  # no water, none predicted -> skipped
    empty_pred = np.zeros((3, 3), dtype=int)
    miss_label = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])  # water, none predicted -> IoU 0
    miss_pred = np.zeros((3, 3), dtype=int)
    mean, used, skipped = mean_chip_iou(
        [PRED, empty_pred, miss_pred], [LABEL, empty_label, miss_label]
    )
    assert (used, skipped) == (2, 1)
    assert math.isclose(mean, (0.4 + 0.0) / 2)


def test_all_chips_skipped_gives_nan():
    z = np.zeros((3, 3), dtype=int)
    mean, used, skipped = mean_chip_iou([z], [z])
    assert math.isnan(mean) and (used, skipped) == (0, 1)


def test_pooled_equals_single_chip_when_one_chip():
    assert pooled_scores([PRED], [LABEL])["iou"] == chip_scores(PRED, LABEL)["iou"]


def _three_chips():
    empty = np.zeros((3, 3), dtype=int)                                  # no water, none predicted
    miss_label = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])             # water, none predicted
    return [PRED, empty, empty], [LABEL, empty, miss_label]


def test_per_chip_table_holds_counts_and_water_share():
    preds, labels = _three_chips()
    t = per_chip_table(preds, labels, ["a", "b", "c"])
    assert t["chip"].tolist() == ["a", "b", "c"]
    assert t["n_valid"].tolist() == [7, 9, 9] and t["n_water"].tolist() == [3, 0, 1]
    assert t[["tp", "fp", "fn"]].values.tolist() == [[2, 2, 1], [0, 0, 0], [0, 0, 1]]
    assert math.isclose(t.loc[0, "water_frac"], 3 / 7) and math.isnan(t.loc[1, "iou"]) and t.loc[2, "iou"] == 0


def test_summarise_agrees_with_the_list_functions():
    preds, labels = _three_chips()
    s = summarise(per_chip_table(preds, labels))
    pooled = pooled_scores(preds, labels)
    mean, used, skipped = mean_chip_iou(preds, labels)
    for key in ("iou", "precision", "recall", "f1", "tp", "fp", "fn"):
        assert s[key] == pooled[key]
    assert math.isclose(s["iou"], 2 / 6)                                 # TP 2, FP 2, FN 1 + 1
    assert (s["mean_chip_iou"], s["chips_in_mean"], s["chips_skipped"]) == (mean, used, skipped) == (0.2, 2, 1)
    assert s["n_chips"] == 3


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    preds = [rng.integers(0, 2, size=(8, 8)) for _ in range(30)]
    labels = [rng.integers(-1, 2, size=(8, 8)) for _ in range(30)]
    t = per_chip_table(preds, labels)
    low, high = bootstrap_iou_ci(t, n_boot=2000, seed=0)
    assert low < summarise(t)["iou"] < high
    assert (low, high) == bootstrap_iou_ci(t, n_boot=2000, seed=0)       # reproducible


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
