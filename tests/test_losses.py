"""Tests for src/losses.py. Run: python -m tests.test_losses  (or pytest tests)

Hand example (same 3x3 as test_metrics, label -1 = no data):
 label          pred (probability)
  1  1  0        0.9 0.2 0.8
  0 -1  1        0.1 0.9 0.7
  0  0 -1        0.9 0.2 0.5      <- the 0.9 and 0.5 sit on -1 pixels and must be ignored
"""
import math

import numpy as np
import tensorflow as tf

from src.losses import WaterIoU, bce_dice_loss, bce_loss, dice_loss

LABEL = np.array([[1, 1, 0], [0, -1, 1], [0, 0, -1]], dtype="float32")
PRED = np.array([[0.9, 0.2, 0.8], [0.1, 0.9, 0.7], [0.9, 0.2, 0.5]], dtype="float32")


def t(a):
    return tf.constant(a[None, :, :, None])  # (1, H, W, 1)


def test_perfect_prediction_gives_zero_dice():
    y = np.array([[1, 1, 0], [0, -1, 1], [0, 0, -1]], dtype="float32")
    perfect = np.where(y == 1, 1.0, 0.0).astype("float32")
    assert float(dice_loss(t(y), t(perfect))) < 1e-5
    assert float(bce_loss(t(y), t(perfect))) < 1e-5


def test_no_water_and_none_predicted_is_not_nan():
    y = np.zeros((3, 3), dtype="float32")
    assert float(dice_loss(t(y), t(y))) < 1e-5


def test_bce_matches_hand_calculation():
    # valid pixels: label 1 with p .9,.2,.7 ; label 0 with p .8,.1,.9,.2
    expected = -(math.log(.9) + math.log(.2) + math.log(.7) + math.log(1 - .8)
                 + math.log(1 - .1) + math.log(1 - .9) + math.log(1 - .2)) / 7
    assert math.isclose(float(bce_loss(t(LABEL), t(PRED))), expected, rel_tol=1e-5)


def test_dice_matches_hand_calculation():
    # sum(p*y) = .9+.2+.7 = 1.8 ; sum(p over valid) = .9+.2+.8+.1+.7+.9+.2 = 3.8 ; sum(y) = 3
    expected = 1 - (2 * 1.8) / (3.8 + 3)
    assert math.isclose(float(dice_loss(t(LABEL), t(PRED))), expected, rel_tol=1e-4)


def test_bce_dice_is_the_sum():
    a, b = float(bce_loss(t(LABEL), t(PRED))), float(dice_loss(t(LABEL), t(PRED)))
    assert math.isclose(float(bce_dice_loss(t(LABEL), t(PRED))), a + b, rel_tol=1e-6)


def test_changing_prediction_on_no_data_pixel_changes_nothing():
    changed = PRED.copy()
    changed[1, 1], changed[2, 2] = 0.0, 1.0  # the two -1 pixels
    for loss in (bce_loss, dice_loss, bce_dice_loss):
        assert math.isclose(float(loss(t(LABEL), t(PRED))), float(loss(t(LABEL), t(changed))), rel_tol=1e-6)
    m1, m2 = WaterIoU(), WaterIoU()
    m1.update_state(t(LABEL), t(PRED))
    m2.update_state(t(LABEL), t(changed))
    assert float(m1.result()) == float(m2.result())


def test_all_no_data_gives_finite_loss():
    y = -np.ones((3, 3), dtype="float32")
    for loss in (bce_loss, dice_loss, bce_dice_loss):
        assert math.isfinite(float(loss(t(y), t(PRED))))


def test_water_iou_matches_hand_value():
    # threshold .5 -> predicted water at (0,0),(0,2),(1,2),(2,0) on valid pixels... see metrics test:
    # valid-pixel TP=(0,0),(1,2)=2 ; FP=(0,2),(2,0)=2 ; FN=(0,1)=1  ->  IoU 2/5
    m = WaterIoU()
    m.update_state(t(LABEL), t(PRED))
    assert math.isclose(float(m.result()), 2 / 5, rel_tol=1e-6)
    m.reset_state()
    assert float(m.result()) == 0.0


def test_water_iou_accumulates_over_batches():
    m = WaterIoU()
    m.update_state(t(LABEL), t(PRED))
    m.update_state(t(LABEL), t(PRED))  # same counts twice -> same ratio
    assert math.isclose(float(m.result()), 2 / 5, rel_tol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
