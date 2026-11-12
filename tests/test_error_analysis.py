"""Tests for src/error_analysis.py: every wrong pixel lands in the right category (synthetic chips)."""
import numpy as np
import pandas as pd

from src import error_analysis as ea

DARK, BRIGHT, T = -30.0, -10.0, -19.95


def blank(value=DARK, size=64):
    return np.zeros((size, size), dtype=np.int8), np.full((size, size), value, dtype=np.float32)


def test_missed_water_thin_edge_interior():
    label, vh = blank()
    label[10:13, :] = 1            # a river 3 pixels wide: thin
    label[30:60, 0:40] = 1         # a lake of 30 x 40 pixels touching the left border
    c = ea.error_categories(np.zeros_like(label, dtype=bool), label, vh, T)
    assert c["miss_thin"] == 3 * 64
    assert c["miss_edge"] == 2 * 40 + 2 * 40 + 2 * (30 - 4)   # 2-pixel rim on the three sides that face land
    assert c["miss_interior"] == 30 * 40 - c["miss_edge"]
    assert c["miss_not_dark"] == c["miss_no_radar"] == 0
    assert sum(v for k, v in c.items() if k.startswith("miss_")) == int((label == 1).sum())


def test_water_touching_the_chip_border_is_not_thin():
    label, vh = blank()
    label[:, :20] = 1              # a wide water body along the left border
    c = ea.error_categories(np.zeros_like(label, dtype=bool), label, vh, T)
    assert c["miss_thin"] == 0


def test_not_dark_and_no_radar_take_precedence():
    label, vh = blank()
    label[10:13, :] = 1
    vh[10:13, :32] = BRIGHT        # half of the river is as bright as land
    vh[10:13, 60:] = np.nan        # 4 columns have no radar value
    c = ea.error_categories(np.zeros_like(label, dtype=bool), label, vh, T)
    assert c["miss_not_dark"] == 3 * 32 and c["miss_no_radar"] == 3 * 4 and c["miss_thin"] == 3 * 28


def test_false_alarm_categories():
    label, vh = blank(BRIGHT)
    label[20:40, 20:40] = 1
    vh[20:40, 20:40] = DARK
    vh[0:5, 0:5] = DARK            # a dark patch of land far from the water
    pred = label.astype(bool).copy()
    pred[18:20, 20:40] = True      # 2 rows just above the lake: shoreline disagreement
    pred[0:5, 0:5] = True          # the dark patch
    pred[50:52, 50:52] = True      # bright land, away from water
    c = ea.error_categories(pred, label, vh, T)
    assert (c["alarm_edge"], c["alarm_dark_land"], c["alarm_other"]) == (40, 25, 4)
    assert sum(v for k, v in c.items() if k.startswith("miss_")) == 0


def test_unlabelled_pixels_are_never_errors():
    label, vh = blank()
    label[:] = -1
    c = ea.error_categories(np.ones_like(label, dtype=bool), label, vh, T)
    assert sum(c.values()) == 0


def test_failure_groups():
    base = {"miss_no_radar": 0, "miss_not_dark": 10, "miss_thin": 50, "miss_edge": 5, "miss_interior": 0,
            "alarm_edge": 5, "alarm_dark_land": 20, "alarm_other": 0}
    assert ea.failure_group({**base, "water_frac": 0.005}) == "little_water"
    assert ea.failure_group({**base, "water_frac": 0.2}) == "miss_thin"
    assert ea.failure_group({**base, "water_frac": 0.2, "miss_thin": 0, "miss_edge": 30}) == "edge"

    table = pd.DataFrame([{"split": "test", "chip": "A_1", "iou": 0.2, "water_frac": 0.2, **base},
                          {"split": "bolivia", "chip": "B_2", "iou": 0.9, "water_frac": 0.3, **base},
                          {"split": "test", "chip": "C_3", "iou": 0.1, "water_frac": 0.001, **base}])
    groups, failing = ea.failure_table(table)
    assert len(failing) == 2 and set(groups["Failure group"]) == {ea.GROUP_LABELS["miss_thin"], ea.GROUP_LABELS["little_water"]}
