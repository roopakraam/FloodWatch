"""Make the report figures from the saved result files (task T27).

    python -m src.make_figures                 # all figures
    python -m src.make_figures --only ablations cost

Every figure is drawn from files under results/, so a changed number means re-running one command,
not redrawing by hand. PNG, 300 dpi, in figures/report/. The same config has the same colour in every
figure (src/viz.py).

    training_curves   loss and water IoU of the three default runs
    ablations         E1 / E2 / E3: mean over seeds, +/- one standard deviation, the individual seeds
    cost              parameters, time per training step, time per predicted chip, GPU memory per encoder
    per_event         pooled IoU per flood event: thresholds against the best U-Net
    qualitative       VH, hand label, Otsu and the best U-Net on six test / Bolivia chips, two of them failures
    water_fraction    per-chip IoU of the best U-Net against the chip's water share

The qualitative figure needs the predicted masks of the best run. They are computed once (this
imports TensorFlow and uses the GPU) and cached in results/<run>/predictions_<split>.npz.
"""
if __package__ in (None, ""):  # allow `python src/make_figures.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import analysis, viz
from src.config import RESULTS_DIR, STUDIES, run_dir, runs_of
from src.data import ROOT, WATER, load_split

FIG_DIR = ROOT / "figures" / "report"
BASELINES = ("Otsu (E0)", "Tuned VH threshold")
STUDY_TITLES = {"E1": "E1 · input channels", "E2": "E2 · loss", "E3": "E3 · encoder"}


def best_run():
    """The U-Net used for the qualitative and error figures: results/best_run.json (chosen on validation)."""
    path = RESULTS_DIR / "best_run.json"
    return json.loads(path.read_text())["run_id"] if path.exists() else "R01"


# ---- predictions of one run, cached -----------------------------------------------------------------------

def predicted_masks(run_id, split):
    """Binary water masks (N, 512, 512) of a run on a split. U-Net masks are cached on first use."""
    data = load_split(split)
    if run_id in analysis.THRESHOLD_RUNS:
        from src.otsu_baseline import predict

        threshold = json.loads((run_dir(run_id) / "metrics.json").read_text())["threshold_db"]
        return predict(data.images, threshold).astype(bool)
    cache = run_dir(run_id) / f"predictions_{split}.npz"
    if cache.exists():
        with np.load(cache) as z:
            if z["names"].tolist() == data.names:
                return np.unpackbits(z["packed"], count=int(np.prod(z["shape"]))).reshape(z["shape"]).astype(bool)
    from src.evaluate import load_run_model, predict_proba
    from src.pipeline import load_arrays
    from src.train import setup_runtime

    cfg = json.loads((run_dir(run_id) / "config.json").read_text())
    setup_runtime(cfg["mixed_precision"])
    model = load_run_model(run_id)
    model.jit_compile = False
    X, _ = load_arrays(split, cfg["channels"])
    masks = predict_proba(model, X, cfg["eval_batch_size"]) >= cfg["threshold"]
    np.savez_compressed(cache, names=np.array(data.names), shape=np.array(masks.shape), packed=np.packbits(masks))
    return masks


# ---- 1. training curves -------------------------------------------------------------------------------------

def fig_training_curves():
    hist = analysis.histories(runs_of("Default"))
    if not hist:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, train, val, title in ((axes[0], "loss", "val_loss", "Loss (BCE + Dice)"),
                                  (axes[1], "water_iou", "val_water_iou", "Water IoU")):
        for i, (run_id, h) in enumerate(hist.items()):
            x = h["epoch"] + 1
            ax.plot(x, h[train], color=viz.GREY, linewidth=1.3, label="training (augmented crops)" if i == 0 else None)
            ax.plot(x, h[val], color=viz.BLUE, linewidth=1.3, label="validation (full chips)" if i == 0 else None)
            best = h["val_water_iou"].idxmax()
            ax.plot(x[best], h.loc[best, val], marker="o", markersize=5.5, color=viz.BLUE, markeredgecolor=viz.SURFACE,
                    markeredgewidth=1.2, linestyle="none", label="epoch that was kept" if i == 0 else None)
        ax.set_title(title)
        ax.set_xlabel("epoch")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return viz.savefig(fig, FIG_DIR / "training_curves.png")


# ---- 2. ablations -------------------------------------------------------------------------------------------

def fig_ablations():
    runs = analysis.runs_table()
    done = runs[runs["Status"] == "Finished"]
    if not set(done["Config"]) - set(BASELINES):
        return None
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 5.0), sharex=True, sharey="row")
    for row, study in enumerate(STUDIES):
        for col, metric in enumerate(("Test IoU", "Bolivia IoU")):
            ax = axes[row, col]
            viz.plot_config_bars(done, STUDIES[study]["configs"], metric, ax)
            for name in BASELINES:  # the two threshold rules as reference lines
                value = pd.to_numeric(done.loc[done["Config"] == name, metric], errors="coerce")
                if len(value):
                    ax.axvline(float(value.iloc[0]), color=viz.CONFIG_COLORS[name], linewidth=1.1, zorder=1)
            ax.set_xlabel("pooled water IoU" if row == 2 else "")
            if row == 0:
                ax.set_title("Test chips" if col == 0 else "Bolivia chips (flood not seen in training)")
        axes[row, 0].annotate(STUDY_TITLES[study], (0.0, 1.0), xycoords="axes fraction", xytext=(-112, 3),
                              textcoords="offset points", fontsize=8.5, fontweight="bold", color=viz.INK, ha="left",
                              va="bottom")
    handles = [plt.Line2D([], [], color=viz.CONFIG_COLORS[n], linewidth=1.3, label=n.replace(" (E0)", " threshold"))
               for n in BASELINES]
    handles += [plt.Line2D([], [], color=viz.INK_2, marker="o", markersize=4, linestyle="none", label="individual seeds"),
                plt.Line2D([], [], color=viz.INK, marker="|", markersize=7, linewidth=1.1, label="mean ± std of 3 seeds")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.025))
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=1.0, h_pad=1.1)
    return viz.savefig(fig, FIG_DIR / "ablations.png")


# ---- 3. cost ------------------------------------------------------------------------------------------------

def fig_cost():
    path = RESULTS_DIR / "cost_table.csv"
    if not path.exists():
        return None
    cost = pd.read_csv(path, index_col=0)
    panels = (("Parameters (millions)", "Parameters", "million", "{:.2f}"),
              ("Training step (ms), benchmark", "Time per training step", "ms (batch of 8 crops)", "{:.0f}"),
              ("Inference per chip (ms), benchmark", "Time per predicted chip", "ms (512×512 pixels)", "{:.1f}"),
              ("Peak GPU memory, training (MB), benchmark", "Peak GPU memory in training", "MB", "{:.0f}"))
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.25), sharey=True)
    names = list(cost.index)
    for ax, (column, title, unit, fmt) in zip(axes, panels):
        values = cost[column].astype(float)
        ax.barh(range(len(names)), values, height=0.5, color=[viz.CONFIG_COLORS[n] for n in names], zorder=2)
        for y, v in enumerate(values):
            ax.text(v + values.max() * 0.03, y, fmt.format(v), va="center", fontsize=8, color=viz.INK)
        ax.set_xlim(0, values.max() * 1.32)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel(unit, fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(len(names) - 0.4, -0.6)
    axes[0].set_yticks(range(len(names)), names)
    fig.tight_layout(w_pad=0.7)
    return viz.savefig(fig, FIG_DIR / "cost.png")


# ---- 4. per event -------------------------------------------------------------------------------------------

def event_iou(run_id, split):
    chips = pd.read_csv(run_dir(run_id) / "per_chip.csv")
    chips = chips[chips["split"] == split]
    g = chips.groupby("event")[["tp", "fp", "fn", "n_water"]].sum()
    g["iou"] = g["tp"] / (g["tp"] + g["fp"] + g["fn"])
    g["chips"] = chips.groupby("event").size()
    return g


def fig_per_event():
    run = best_run()
    if not (run_dir(run) / "metrics.json").exists() or "test" not in json.loads((run_dir(run) / "metrics.json").read_text()):
        return None
    series = (("R00", "Otsu threshold", viz.CONFIG_COLORS["Otsu (E0)"]),
              ("R00b", "tuned threshold", viz.CONFIG_COLORS["Tuned VH threshold"]),
              (run, f"U-Net ({run})", viz.BLUE))
    frames = {rid: pd.concat([event_iou(rid, "test"), event_iou(rid, "bolivia")]) for rid, _, _ in series}
    unet = frames[run].sort_values("iou")
    order = [e for e in unet.index if e != "Bolivia"] + ["Bolivia"]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for y, event in enumerate(order):
        values = [frames[rid].loc[event, "iou"] for rid, _, _ in series]
        ax.plot([min(values), max(values)], [y, y], color=viz.AXIS, linewidth=1.2, zorder=1)
        for (rid, label, color), v in zip(series, values):
            ax.scatter(v, y, s=44, color=color, edgecolor=viz.SURFACE, linewidth=1.2, zorder=3,
                       label=label if y == 0 else None)
    labels = [f"{e}  ({int(unet.loc[e, 'chips'])} chips)" for e in order]
    ax.set_yticks(range(len(order)), labels)
    ax.axhline(len(order) - 1.5, color=viz.AXIS, linewidth=0.8)
    ax.annotate("held-out flood", (0.995, len(order) - 1.42), xycoords=("axes fraction", "data"), ha="right", va="top",
                fontsize=7.5, color=viz.MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("pooled water IoU of the event's chips")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=1)
    fig.tight_layout()
    return viz.savefig(fig, FIG_DIR / "per_event.png")


# ---- 5. qualitative grid ------------------------------------------------------------------------------------

def pick_qualitative_chips(run):
    """Six chips by a fixed rule, from the best U-Net's own per-chip scores:
    two failures (lowest IoU among test chips with at least 2% water), three test chips at the 50th, 75th
    and 95th percentile of IoU, and the median Bolivia chip."""
    chips = pd.read_csv(run_dir(run) / "per_chip.csv")
    test = chips[(chips["split"] == "test") & (chips["water_frac"] >= 0.02)].sort_values("iou").reset_index(drop=True)
    bolivia = chips[(chips["split"] == "bolivia") & (chips["water_frac"] >= 0.02)].sort_values("iou").reset_index(drop=True)
    idx = [0, 1] + [int(round(q * (len(test) - 1))) for q in (0.50, 0.75, 0.95)]
    picks = [("test", test.loc[i, "chip"]) for i in idx] + [("bolivia", bolivia.loc[len(bolivia) // 2, "chip"])]
    return picks


def fig_qualitative():
    run = best_run()
    if not (run_dir(run) / "best.keras").exists() or "test" not in json.loads((run_dir(run) / "metrics.json").read_text()):
        return None
    picks = pick_qualitative_chips(run)
    data = {s: load_split(s) for s in ("test", "bolivia")}
    masks = {(rid, s): predicted_masks(rid, s) for rid in ("R00", run) for s in ("test", "bolivia")}
    scores = {rid: pd.read_csv(run_dir(rid) / "per_chip.csv").set_index("chip") for rid in ("R00", run)}

    fig, axes = plt.subplots(len(picks), 4, figsize=(7.0, 1.78 * len(picks)))
    for row, (split, name) in enumerate(picks):
        i = data[split].names.index(name)
        image, label = data[split].images[i], data[split].labels[i]
        share = (label == WATER).sum() / max((label != -1).sum(), 1)
        viz.show_db(axes[row, 0], image[1], "VH")
        viz.show_label(axes[row, 1], label)
        viz.show_errors(axes[row, 2], masks[("R00", split)][i], label)
        viz.show_errors(axes[row, 3], masks[(run, split)][i], label)
        tag = "failure" if row < 2 else ("Bolivia" if split == "bolivia" else "test")
        axes[row, 0].set_ylabel(f"{name.replace('_', ' ')}\n{tag}, {share:.0%} water", fontsize=7.5, color=viz.INK_2)
        axes[row, 2].set_xlabel(f"IoU {scores['R00'].loc[name, 'iou']:.2f}", fontsize=7.5, color=viz.INK_2, labelpad=2)
        axes[row, 3].set_xlabel(f"IoU {scores[run].loc[name, 'iou']:.2f}", fontsize=7.5, color=viz.INK_2, labelpad=2)
    for ax, title in zip(axes[0], ("VH (dB)", "hand label", "Otsu threshold", f"U-Net ({run})")):
        ax.set_title(title, fontsize=8.5, fontweight="bold", color=viz.INK, loc="center", pad=4)
    fig.legend(handles=viz.error_legend_handles(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.012), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=0.5, w_pad=0.3)
    return viz.savefig(fig, FIG_DIR / "qualitative.png")


# ---- 6. IoU against the amount of water ----------------------------------------------------------------------

def fig_water_fraction():
    run = best_run()
    path = run_dir(run) / "per_chip.csv"
    if not path.exists():
        return None
    chips = pd.read_csv(path)
    chips = chips[chips["split"].isin(["test", "bolivia"]) & (chips["n_water"] > 0)]
    if chips.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for split, color, label in (("test", viz.BLUE, "test chips"), ("bolivia", viz.ORANGE, "Bolivia chips")):
        c = chips[chips["split"] == split]
        ax.scatter(c["water_frac"].clip(lower=2e-5), c["iou"], s=24, color=color, edgecolor=viz.SURFACE, linewidth=0.7,
                   label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("water share of the chip's labelled pixels (log scale)")
    ax.set_ylabel(f"per-chip IoU, U-Net ({run})")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return viz.savefig(fig, FIG_DIR / "iou_vs_water_fraction.png")


FIGURES = {"training_curves": fig_training_curves, "ablations": fig_ablations, "cost": fig_cost,
           "per_event": fig_per_event, "qualitative": fig_qualitative, "water_fraction": fig_water_fraction}


def main():
    parser = argparse.ArgumentParser(description="Make the report figures from results/.")
    parser.add_argument("--only", nargs="*", choices=list(FIGURES), help="draw only these figures")
    args = parser.parse_args()
    plt.switch_backend("Agg")  # the script only writes files
    viz.use_style()
    for name in args.only or FIGURES:
        path = FIGURES[name]()
        print(f"{name:16s} -> {path.relative_to(ROOT) if path else 'skipped (results not available yet)'}")
        plt.close("all")


if __name__ == "__main__":
    main()
