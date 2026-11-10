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

from src.metrics import chip_scores, confusion_counts, mean_chip_iou, pooled_scores

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
