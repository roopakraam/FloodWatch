"""Why does a model get pixels wrong? Rule-based categories for every error pixel (task T26).

A results table says how well a model does; this module says where the errors are. Every wrong pixel
of a prediction is put into exactly one category by fixed, checkable rules (no judgement by eye):

Missed water (label = water, prediction = not water), first matching rule wins:
    no_radar   the chip has no radar value at this pixel (nothing to see)
    not_dark   VH is at or above the Otsu threshold: in the radar image this water looks like land
    thin       the pixel belongs to a thin or small water body (it vanishes when the labelled water is
               opened with a 5x5 square, i.e. the structure is narrower than about 50 m)
    edge       within 2 pixels of the boundary of the labelled water (shoreline disagreement)
    interior   dark, well inside a larger water body: a plain miss

False alarms (label = not water, prediction = water):
    edge       within 2 pixels of labelled water (shoreline disagreement)
    dark_land  VH below the Otsu threshold, away from labelled water: land that looks like water in the
               radar image (smooth surfaces, radar shadow, wet soil, or water the labeller did not mark)
    other      neither dark nor near water

A chip counts as a failure when its IoU is below 0.5. Failing chips are grouped by a rule as well:
"little water" if under 1% of the chip is labelled water, otherwise by the category that holds most
of the chip's error pixels.
"""
import numpy as np
import pandas as pd
from scipy import ndimage

from src.data import NO_DATA, NON_WATER, WATER, event_of

MISS_CATEGORIES = ("no_radar", "not_dark", "thin", "edge", "interior")
ALARM_CATEGORIES = ("edge", "dark_land", "other")
CATEGORY_LABELS = {
    "miss_no_radar": "missed water: no radar data",
    "miss_not_dark": "missed water: not dark in the radar image",
    "miss_thin": "missed water: thin or small water body",
    "miss_edge": "missed water: shoreline",
    "miss_interior": "missed water: dark, inside a larger water body",
    "alarm_edge": "false alarm: shoreline",
    "alarm_dark_land": "false alarm: dark land",
    "alarm_other": "false alarm: other",
}
GROUP_LABELS = {
    "little_water": "little or no water (under 1% of the chip)",
    "miss_not_dark": "water that is not dark in the radar image",
    "miss_thin": "thin rivers and small water bodies",
    "edge": "edges of water bodies",
    "alarm_dark_land": "dark land taken for water",
    "miss_interior": "dark water missed",
    "miss_no_radar": "no radar data",
    "alarm_other": "other false alarms",
}
OPEN = np.ones((5, 5), dtype=bool)   # structures narrower than 5 pixels (50 m) count as thin
EDGE_PX = 2
FAILURE_IOU = 0.5
LITTLE_WATER = 0.01


def _opened(mask):
    """Binary opening with the 5x5 square. The mask is padded by repeating its border first, so that water
    touching the edge of the chip is not mistaken for a thin structure."""
    padded = np.pad(mask, 4, mode="edge")
    return ndimage.binary_opening(padded, structure=OPEN)[4:-4, 4:-4]


def error_categories(pred, label, vh, dark_threshold):
    """Pixel counts per error category for one chip. `vh` is the raw VH band in dB (NaN = no radar)."""
    pred = np.asarray(pred).astype(bool)
    valid = label != NO_DATA
    water = label == WATER
    miss = water & ~pred
    alarm = (label == NON_WATER) & pred

    no_radar = ~np.isfinite(vh)
    with np.errstate(invalid="ignore"):
        dark = vh < dark_threshold                      # NaN < t is False
    thin = water & ~_opened(water)
    near_water = ndimage.binary_dilation(water, iterations=EDGE_PX)
    near_land = ndimage.binary_dilation(valid & ~water, iterations=EDGE_PX)

    counts, rest = {}, miss.copy()
    for name, rule in (("no_radar", no_radar), ("not_dark", ~dark), ("thin", thin), ("edge", near_land)):
        hit = rest & rule
        counts[f"miss_{name}"] = int(hit.sum())
        rest &= ~hit
    counts["miss_interior"] = int(rest.sum())

    rest = alarm.copy()
    for name, rule in (("edge", near_water), ("dark_land", dark)):
        hit = rest & rule
        counts[f"alarm_{name}"] = int(hit.sum())
        rest &= ~hit
    counts["alarm_other"] = int(rest.sum())
    return counts


def chip_error_table(masks, data, dark_threshold, split):
    """One row per chip: IoU, pixel counts and the error pixels per category."""
    rows = []
    for name, pred, label, image in zip(data.names, masks, data.labels, data.images):
        pred = np.asarray(pred).astype(bool)
        valid, water = label != NO_DATA, label == WATER
        tp = int((pred & water).sum())
        fp = int((pred & valid & ~water).sum())
        fn = int((~pred & water).sum())
        rows.append({"split": split, "chip": name, "event": event_of(name),
                     "water_frac": water.sum() / max(valid.sum(), 1), "n_water": int(water.sum()),
                     "tp": tp, "fp": fp, "fn": fn, "iou": tp / (tp + fp + fn) if tp + fp + fn else np.nan,
                     **error_categories(pred, label, image[1], dark_threshold)})
    return pd.DataFrame(rows)


def error_budget(table):
    """Where the wrong pixels are: each category's pixel count and its share of all error pixels."""
    cols = [f"miss_{c}" for c in MISS_CATEGORIES] + [f"alarm_{c}" for c in ALARM_CATEGORIES]
    counts = table[cols].sum()
    out = pd.DataFrame({"pixels": counts, "share of all errors": counts / counts.sum()})
    out.index = [CATEGORY_LABELS[c] for c in cols]
    return out


def failure_group(row):
    """The failure group of one failing chip (see the module docstring)."""
    if row["water_frac"] < LITTLE_WATER:
        return "little_water"
    cats = {"miss_no_radar": row["miss_no_radar"], "miss_not_dark": row["miss_not_dark"], "miss_thin": row["miss_thin"],
            "edge": row["miss_edge"] + row["alarm_edge"], "miss_interior": row["miss_interior"],
            "alarm_dark_land": row["alarm_dark_land"], "alarm_other": row["alarm_other"]}
    return max(cats, key=cats.get)


def failure_table(table, reference=None):
    """Failing chips (IoU < 0.5) counted per group, with the worst chip of each group as the example.

    `reference` is an optional chip table of another method (e.g. Otsu) for the same chips; its mean IoU
    on each group's chips is added for comparison.
    """
    failing = table[table["iou"] < FAILURE_IOU].copy()
    failing["group"] = failing.apply(failure_group, axis=1)
    ref = reference.set_index("chip")["iou"] if reference is not None else None
    rows = []
    for group, chips in failing.groupby("group"):
        worst = chips.sort_values("iou").iloc[0]
        row = {"Failure group": GROUP_LABELS[group], "Test chips": int((chips["split"] == "test").sum()),
               "Bolivia chips": int((chips["split"] == "bolivia").sum()), "Mean IoU": chips["iou"].mean(),
               "Example chip": worst["chip"], "Example IoU": worst["iou"]}
        if ref is not None:
            row["Otsu mean IoU, same chips"] = ref.reindex(chips["chip"]).mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["_n"] = out["Test chips"] + out["Bolivia chips"]
        out = out.sort_values("_n", ascending=False).drop(columns="_n").reset_index(drop=True)
    return out, failing
