"""Dataset checks and statistics (tasks T07 and T09).

Everything that feeds a design decision is computed on the TRAINING split only: the water share
(-> why Dice is tested), the dB histograms (-> why a threshold is a fair baseline), the NaN / no-data
counts (-> what must be masked). The per-split summary is label bookkeeping for the report's dataset
table; it is not used to choose anything.

Run from the project root:  python -m src.data_stats

Writes
    results/data_integrity.json    file and split counts, consistency checks                      (T07)
    results/data_stats.csv         one row per training chip: valid / water / no-data / NaN pixels  (T09)
    results/data_summary.json      training-set totals                                            (T09)
    results/db_histograms.csv      VV / VH histograms of training pixels, water vs not water      (T09)
    results/split_summary.csv      one row per split: chips, events, pixel shares (description only)
    figures/eda/sample_chips.png, figures/eda/hist_vv_vh_water.png, figures/eda/water_fraction_per_chip.png
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

from src.data import (BANDS, CHIP_SIZE, LABEL_DIR, NO_DATA, NON_WATER, ROOT, S1_DIR, SPLIT_SIZES, SPLITS, WATER,
                      chip_name, event_of, list_split, load_split)
from src.preprocess import CLIP_DB

RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures" / "eda"

HIST_EDGES = np.arange(CLIP_DB[0], CLIP_DB[1] + 0.25, 0.5)  # 0.5 dB bins over the clip range
CLASSES = {"water": WATER, "not_water": NON_WATER}
WATER_BINS = [("no water", -1, 0.0), ("up to 1%", 0.0, 0.01), ("1 to 5%", 0.01, 0.05), ("5 to 25%", 0.05, 0.25),
              ("25 to 50%", 0.25, 0.50), ("over 50%", 0.50, 1.0)]  # (label, exclusive low, inclusive high)


# ---- T07: are the files and splits what they should be? ------------------------------------------------

def check_integrity():
    """Counts and consistency checks. Raises AssertionError if anything is off."""
    s1_files = sorted(p.name for p in S1_DIR.glob("*_S1Hand.tif"))
    label_files = sorted(p.name for p in LABEL_DIR.glob("*_LabelHand.tif"))
    pairs = {s: list_split(s) for s in SPLITS}
    names = {s: [chip_name(a) for a, _ in pairs[s]] for s in SPLITS}
    all_names = [n for s in SPLITS for n in names[s]]

    missing = [str(p.name) for s in SPLITS for pair in pairs[s] for p in pair if not p.exists()]
    mismatched = [f"{a.name} / {b.name}" for s in SPLITS for a, b in pairs[s] if chip_name(a) != chip_name(b)]
    duplicates = sorted(n for n, c in Counter(all_names).items() if c > 1)
    unlisted = sorted(set(chip_name(f) for f in s1_files) - set(all_names))

    report = {
        "s1_files_on_disk": len(s1_files),
        "label_files_on_disk": len(label_files),
        "chips_per_split": {s: len(names[s]) for s in SPLITS},
        "chips_expected_per_split": SPLIT_SIZES,
        "chips_in_all_splits": len(all_names),
        "unique_chips_in_all_splits": len(set(all_names)),
        "files_listed_but_missing": missing,
        "pairs_with_different_chip_ids": mismatched,
        "chips_in_more_than_one_split": duplicates,
        "chips_on_disk_not_in_any_split": unlisted,
        "events_per_split": {s: dict(sorted(Counter(event_of(n) for n in names[s]).items())) for s in SPLITS},
    }
    assert report["s1_files_on_disk"] == report["label_files_on_disk"] == 446, report
    assert report["chips_per_split"] == SPLIT_SIZES, report["chips_per_split"]
    assert not (missing or mismatched or duplicates or unlisted), report
    assert set(report["events_per_split"]["bolivia"]) == {"Bolivia"}
    assert "Bolivia" not in {e for s in ("train", "val", "test") for e in report["events_per_split"][s]}
    return report


# ---- T09: per-chip and pooled statistics ----------------------------------------------------------------

def chip_table(split):
    """One row per chip of `split`: pixel counts by label class and missing-data counts."""
    data = load_split(split)
    rows = []
    for name, image, label in zip(data.names, data.images, data.labels):
        valid = label != NO_DATA
        nan = np.isnan(image).any(axis=0)
        n_valid, n_water = int(valid.sum()), int((label == WATER).sum())
        rows.append({
            "chip": name,
            "event": event_of(name),
            "n_pixels": label.size,
            "n_valid": n_valid,
            "n_water": n_water,
            "water_frac": n_water / n_valid if n_valid else np.nan,
            "n_label_nodata": int((~valid).sum()),
            "n_nan_image": int(nan.sum()),
            "n_nan_image_valid_label": int((nan & valid).sum()),
            "vv_mean_db": float(np.nanmean(image[0])) if not nan.all() else np.nan,
            "vh_mean_db": float(np.nanmean(image[1])) if not nan.all() else np.nan,
        })
    return pd.DataFrame(rows)


def db_histograms(split="train"):
    """Tidy histogram table: band, class, bin edges, pixel count.

    0.5 dB bins over the clip range, plus one open bin on each side (bin_left = -inf, bin_right = inf)
    holding the pixels that the preprocessing clips to -35 dB or to 5 dB.
    """
    edges = np.concatenate([[-np.inf], HIST_EDGES, [np.inf]])
    data = load_split(split)
    counts = {(b, c): np.zeros(len(edges) - 1, dtype=np.int64) for b in BANDS for c in CLASSES}
    for image, label in zip(data.images, data.labels):
        for bi, band in enumerate(BANDS):
            finite = ~np.isnan(image[bi])
            for cname, cval in CLASSES.items():
                counts[(band, cname)] += np.histogram(image[bi][finite & (label == cval)], bins=edges)[0]
    rows = [{"band": band, "class": cname, "bin_left": edges[i], "bin_right": edges[i + 1], "count": int(c[i])}
            for (band, cname), c in counts.items() for i in range(len(c))]
    return pd.DataFrame(rows)


def clipped_share(hist, band, cname):
    """(share of the class's pixels below the clip range, share above it)."""
    h = hist[(hist["band"] == band) & (hist["class"] == cname)]
    total = h["count"].sum()
    return (float(h.loc[np.isneginf(h["bin_left"]), "count"].sum() / total),
            float(h.loc[np.isposinf(h["bin_right"]), "count"].sum() / total))


def water_bins(table):
    """How many chips fall in each water-share class (ordered from none to mostly water)."""
    frac = table["water_frac"]
    return pd.Series({label: int(((frac > lo) & (frac <= hi)).sum()) for label, lo, hi in WATER_BINS}, name="chips")


def summarise_split(table, split):
    """Pooled pixel statistics of one split from its chip table."""
    n_px, n_valid, n_water = (int(table[c].sum()) for c in ("n_pixels", "n_valid", "n_water"))
    return {
        "split": split,
        "n_chips": int(len(table)),
        "n_events": int(table["event"].nunique()),
        "n_pixels": n_px,
        "n_valid_px": n_valid,
        "n_water_px": n_water,
        "water_share_of_valid_px": n_water / n_valid,
        "n_label_nodata_px": int(table["n_label_nodata"].sum()),
        "label_nodata_share": float(table["n_label_nodata"].sum() / n_px),
        "n_nan_image_px": int(table["n_nan_image"].sum()),
        "nan_image_share": float(table["n_nan_image"].sum() / n_px),
        "n_nan_image_px_with_valid_label": int(table["n_nan_image_valid_label"].sum()),
        "chips_no_water": int((table["water_frac"] == 0).sum()),
        "chips_water_below_1pct": int((table["water_frac"] < 0.01).sum()),
        "chips_water_below_5pct": int((table["water_frac"] < 0.05).sum()),
        "median_chip_water_frac": float(table["water_frac"].median()),
        "chips_without_any_sar_data": int((table["n_nan_image"] == table["n_pixels"]).sum()),
    }


def clip_counts(split="train"):
    """How many finite training pixels fall outside the clip range (they are clipped, not removed)."""
    images = load_split(split).images
    finite = np.isfinite(images)
    return {"n_finite_values": int(finite.sum()),
            "n_below_clip": int((images[finite] < CLIP_DB[0]).sum()),
            "n_above_clip": int((images[finite] > CLIP_DB[1]).sum()),
            "min_db": float(np.nanmin(images)), "max_db": float(np.nanmax(images))}


# ---- figures ----------------------------------------------------------------------------------------------

def plot_db_histograms(hist, threshold=None, axes=None):
    """VV and VH distributions of training pixels, water vs not water (each class normalised to 1)."""
    import matplotlib.pyplot as plt

    from src import viz

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
    for ax, band in zip(axes, BANDS):
        for cname, color, label in (("not_water", viz.GREY, "not water"), ("water", viz.BLUE, "water")):
            h = hist[(hist["band"] == band) & (hist["class"] == cname)]
            inside = h[np.isfinite(h["bin_left"]) & np.isfinite(h["bin_right"])]
            centers = (inside["bin_left"] + inside["bin_right"]) / 2
            density = inside["count"] / h["count"].sum() / (inside["bin_right"] - inside["bin_left"])
            ax.plot(centers, density, color=color, label=label)
            ax.fill_between(centers, density, color=color, alpha=0.10, linewidth=0)
        ax.set_title(f"{band} backscatter")
        ax.set_xlabel("dB")
        ax.set_xlim(*CLIP_DB)
        ax.set_ylim(bottom=0)
    for ax, band in zip(axes, BANDS):  # the pixels below -35 dB are clipped; say how many instead of drawing a spike
        below = clipped_share(hist, band, "water")[0]
        if below >= 0.005:
            ax.annotate(f"{below:.1%} of water\npixels lie below\n-35 dB (not drawn)", (CLIP_DB[0] + 0.3, 0.004),
                        xytext=(0.04, 0.74), textcoords="axes fraction", va="center", fontsize=8, color=viz.INK_2,
                        arrowprops={"arrowstyle": "-", "color": viz.MUTED, "linewidth": 0.8, "shrinkA": 2,
                                    "shrinkB": 0, "relpos": (0.0, 0.0)})
    if threshold is not None:
        axes[1].axvline(threshold, color=viz.INK, linewidth=1)
        axes[1].annotate(f"Otsu threshold\n{threshold:.2f} dB", (threshold, axes[1].get_ylim()[1] * 0.97),
                         xytext=(5, 0), textcoords="offset points", va="top", fontsize=8, color=viz.INK_2)
    axes[0].set_ylabel("share of the class's pixels per dB")
    axes[0].legend(loc="upper left")
    return axes


def plot_water_fraction(table, ax=None):
    """Number of training chips per water-share class."""
    import matplotlib.pyplot as plt

    from src import viz

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.0))
    bins = water_bins(table)
    bars = ax.bar(bins.index, bins.values, width=0.45, color=viz.BLUE)
    ax.bar_label(bars, padding=2, fontsize=8, color=viz.INK)
    ax.set_ylabel("training chips")
    ax.set_xlabel("water share of the chip's labelled pixels")
    ax.set_title("Most training chips contain little or no water")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, bins.max() * 1.15)
    return ax


SAMPLE_EVENTS = ("Ghana", "India", "Mekong", "Paraguay", "Spain", "USA")


def pick_sample_chips(table, events=SAMPLE_EVENTS, target_water=0.20):
    """One chip per event: the one whose water share is closest to `target_water` and that has no missing
    SAR pixels, so both classes are visible. A fixed rule, so the figure is reproducible."""
    picks = []
    for event in events:
        t = table[(table["event"] == event) & (table["n_nan_image"] == 0)]
        picks.append(t.loc[(t["water_frac"] - target_water).abs().idxmin(), "chip"])
    return picks


def plot_sample_chips(names, split="train"):
    """VV, VH and the hand label of the given chips, one column per chip. Fixed dB ranges for all chips."""
    import matplotlib.pyplot as plt

    from src import viz

    data = load_split(split)
    index = {n: i for i, n in enumerate(data.names)}
    fig, axes = plt.subplots(3, len(names), figsize=(1.75 * len(names), 5.9), squeeze=False)
    for col, name in enumerate(names):
        image, label = data.images[index[name]], data.labels[index[name]]
        share = (label == WATER).sum() / max((label != NO_DATA).sum(), 1)
        viz.show_db(axes[0, col], image[0], "VV", title=name.replace("_", " "))
        viz.show_db(axes[1, col], image[1], "VH")
        viz.show_label(axes[2, col], label, title=f"{share:.0%} water")
    for row, text in enumerate(("VV (dB)", "VH (dB)", "hand label")):
        axes[row, 0].set_ylabel(text, fontsize=9, color=viz.INK_2)
    fig.legend(handles=viz.label_legend_handles(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    return fig


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import viz

    viz.use_style()
    RESULTS.mkdir(exist_ok=True)

    integrity = check_integrity()
    (RESULTS / "data_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    print("integrity OK:", integrity["chips_per_split"])

    train = chip_table("train")
    train.to_csv(RESULTS / "data_stats.csv", index=False)

    summary = {**summarise_split(train, "train"), "clip_db": list(CLIP_DB), **clip_counts("train"),
               "chips_per_water_class": water_bins(train).to_dict()}
    (RESULTS / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    hist = db_histograms("train")
    hist.to_csv(RESULTS / "db_histograms.csv", index=False)

    splits = pd.DataFrame([summarise_split(train if s == "train" else chip_table(s), s) for s in SPLITS])
    splits.to_csv(RESULTS / "split_summary.csv", index=False)

    plot_db_histograms(hist)
    plt.gcf().tight_layout()
    viz.savefig(plt.gcf(), FIG_DIR / "hist_vv_vh_water.png")
    plot_water_fraction(train)
    plt.gcf().tight_layout()
    viz.savefig(plt.gcf(), FIG_DIR / "water_fraction_per_chip.png")
    viz.savefig(plot_sample_chips(pick_sample_chips(train)), FIG_DIR / "sample_chips.png")

    print(f"train: {summary['n_chips']} chips | water share of labelled pixels {summary['water_share_of_valid_px']:.2%} | "
          f"{summary['chips_no_water']} chips without water | no-data label share {summary['label_nodata_share']:.2%}")
    print(splits[["split", "n_chips", "n_events", "water_share_of_valid_px", "label_nodata_share", "nan_image_share"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
