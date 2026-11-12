"""Shared plotting helpers: one style, one colour per config, one colour per map class.

Every notebook and every report figure draws through these helpers, so the same config has the same
colour in every figure and all radar images use the same fixed dB range (chips are comparable).

Colours come from a categorical palette that was checked for colour-vision deficiency (protan / deutan)
for the pairs that actually sit next to each other in the figures:
    E1  VH only (orange)  | VV+VH (aqua)     | Default (blue)
    E2  BCE (yellow)      | Dice (red)       | Default (blue)
    E3  Scratch (green)   | Default (blue)   | MobileNetV2 + ImageNet (magenta)
The two threshold baselines are drawn in neutral greys: they are context, not compared models.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

# ---- ink and chrome --------------------------------------------------------------------------------
INK = "#0b0b0b"         # titles, values
INK_2 = "#52514e"       # axis labels, legends
MUTED = "#898781"       # tick labels, notes
GRID = "#e1e0d9"        # hairline grid
AXIS = "#c3c2b7"        # baseline / spines
SURFACE = "#ffffff"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, RED, GREY = "#e87ba4", "#008300", "#e34948", "#898781"

# ---- one colour per config, used in every figure -----------------------------------------------------
CONFIG_COLORS = {
    "Otsu (E0)": GREY,
    "Tuned VH threshold": INK_2,
    "Default": BLUE,
    "VH only": ORANGE,
    "VV+VH": AQUA,
    "BCE loss": YELLOW,
    "Dice loss": RED,
    "Scratch U-Net": GREEN,
    "MobileNetV2 + ImageNet": MAGENTA,
}

# ---- map classes ------------------------------------------------------------------------------------
WATER_COLOR = BLUE
LAND_COLOR = "#f0efec"
NODATA_COLOR = "#c3c2b7"
FALSE_ALARM_COLOR = RED      # predicted water, label says land
MISSED_COLOR = YELLOW        # label says water, not predicted

LABEL_CMAP = ListedColormap([NODATA_COLOR, LAND_COLOR, WATER_COLOR])            # -1, 0, 1
LABEL_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], LABEL_CMAP.N)
ERROR_CMAP = ListedColormap([NODATA_COLOR, LAND_COLOR, WATER_COLOR, FALSE_ALARM_COLOR, MISSED_COLOR])
ERROR_NORM = BoundaryNorm(np.arange(-0.5, 5.5), ERROR_CMAP.N)

# Fixed display ranges in dB, the same for every chip.
DB_RANGE = {"VV": (-25.0, 0.0), "VH": (-32.0, -7.0)}

DPI = 300


def use_style():
    """Quiet, consistent matplotlib defaults: hairline solid grid, no top/right spines, dark-grey text."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "figure.dpi": 110, "savefig.dpi": DPI, "savefig.bbox": "tight",
    })


def savefig(fig, path, dpi=DPI):
    """Save a figure (300 dpi PNG by default), creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ---- image panels -----------------------------------------------------------------------------------

def _image_axis(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, fontweight="normal", color=INK_2, loc="center", pad=3)


def show_db(ax, band, kind="VH", title=None):
    """A radar band in dB, grey scale with the fixed range for that band. NaN (no SAR data) shows as the
    no-data grey."""
    lo, hi = DB_RANGE[kind]
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(NODATA_COLOR)
    im = ax.imshow(np.ma.masked_invalid(band), cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    _image_axis(ax, title)
    return im


def show_label(ax, label, title=None):
    """A label or a binary mask: grey = no data, light = not water, blue = water."""
    ax.imshow(label, cmap=LABEL_CMAP, norm=LABEL_NORM, interpolation="nearest")
    _image_axis(ax, title)


def error_classes(pred, label):
    """0 no data, 1 correct land, 2 correct water, 3 false alarm, 4 missed water."""
    pred, label = np.asarray(pred).astype(bool), np.asarray(label)
    out = np.zeros(label.shape, dtype=np.int8)
    valid = label != -1
    out[valid & ~pred & (label == 0)] = 1
    out[valid & pred & (label == 1)] = 2
    out[valid & pred & (label == 0)] = 3
    out[valid & ~pred & (label == 1)] = 4
    return out


def show_errors(ax, pred, label, title=None):
    """Prediction compared with the label: blue = water found, red = false alarm, yellow = missed water."""
    ax.imshow(error_classes(pred, label), cmap=ERROR_CMAP, norm=ERROR_NORM, interpolation="nearest")
    _image_axis(ax, title)


def label_legend_handles():
    return [Patch(facecolor=WATER_COLOR, label="water"),
            Patch(facecolor=LAND_COLOR, edgecolor=AXIS, linewidth=0.5, label="not water"),
            Patch(facecolor=NODATA_COLOR, label="no data")]


def error_legend_handles():
    return [Patch(facecolor=WATER_COLOR, label="water found"),
            Patch(facecolor=FALSE_ALARM_COLOR, label="false alarm"),
            Patch(facecolor=MISSED_COLOR, label="missed water"),
            Patch(facecolor=LAND_COLOR, edgecolor=AXIS, linewidth=0.5, label="not water"),
            Patch(facecolor=NODATA_COLOR, label="no data")]


# ---- training curves --------------------------------------------------------------------------------

def plot_history(history, axes=None, title=None, best_epoch=None):
    """Loss and water IoU against the epoch for one run. Training in grey, validation in blue."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 3))
    epochs = history["epoch"] + 1
    for ax, train, val, name in ((axes[0], "loss", "val_loss", "Loss"),
                                 (axes[1], "water_iou", "val_water_iou", "Water IoU")):
        ax.plot(epochs, history[train], color=GREY, label="training (augmented crops)")
        ax.plot(epochs, history[val], color=BLUE, label="validation (full chips)")
        if best_epoch is not None:
            ax.axvline(best_epoch, color=AXIS, linewidth=1)
        ax.set_xlabel("epoch")
        ax.set_title(name if not title else f"{title}: {name.lower()}")
    if best_epoch is not None:
        axes[1].annotate(f"best epoch {best_epoch}", (best_epoch, axes[1].get_ylim()[0]), xytext=(4, 4),
                         textcoords="offset points", fontsize=8, color=MUTED)
    axes[1].legend(loc="lower right")
    return axes


# ---- ablation studies -------------------------------------------------------------------------------

def plot_val_curves(histories_by_config, axes=None, ylim=None):
    """Validation water IoU against the epoch, one panel per config, one line per seed.

    `histories_by_config` maps a config name to {run_id: history DataFrame}.
    """
    n = len(histories_by_config)
    if axes is None:
        _, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.0), sharey=True, squeeze=False)
        axes = axes[0]
    for ax, (config, runs) in zip(axes, histories_by_config.items()):
        color = CONFIG_COLORS[config]
        for run_id, h in runs.items():
            ax.plot(h["epoch"] + 1, h["val_water_iou"], color=color, linewidth=1.4, alpha=0.9)
            best = h["val_water_iou"].idxmax()
            ax.plot(h.loc[best, "epoch"] + 1, h.loc[best, "val_water_iou"], marker="o", markersize=5.5, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="none")
        ax.set_title(config)
        ax.set_xlabel("epoch")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel("validation water IoU")
    return axes


def plot_config_bars(runs, configs, metric, ax, baseline=None, xlim=(0, 1), label_fmt="{:.3f}"):
    """Horizontal bars = mean over seeds, whisker = +/- one std, dots = the individual seeds.

    `runs` is analysis.runs_table() (finished runs). Bars start at zero, so small differences look small;
    the label at each bar gives the exact mean and std.
    """
    import pandas as pd

    baselines = [baseline] if isinstance(baseline, str) else list(baseline or [])
    names = baselines + list(configs)
    for y, name in enumerate(names):
        values = pd.to_numeric(runs.loc[runs["Config"] == name, metric], errors="coerce").dropna().to_numpy()
        if not len(values):
            continue
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else None
        ax.barh(y, mean, height=0.5, color=CONFIG_COLORS[name], zorder=2)
        if std is not None:
            ax.errorbar(mean, y, xerr=std, color=INK, linewidth=1.1, capsize=3, capthick=1.1, zorder=4)
            ax.scatter(values, [y + 0.33] * len(values), s=16, color=INK_2, edgecolor=SURFACE, linewidth=0.8, zorder=5)
        text = label_fmt.format(mean) + (f" ± {std:.3f}" if std is not None else "")
        ax.text(max(values.max(), mean + (std or 0)) + 0.015, y, text, va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(metric)
    return ax
