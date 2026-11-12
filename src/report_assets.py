"""Turn the saved results into the numbers and tables that the report pulls in (tasks T28 to T30, T32).

    python -m src.report_assets

Writes
    report/generated/numbers.json     every number quoted in the report text, as a formatted string
    report/generated/sources.json     for each number: the result file (and field) it was read from
    report/generated/tables/*.md      the report's tables

The report source (report/report.md and report/sections/*.md) contains placeholders such as
{{default_test_iou_pm}} instead of typed numbers. src/build_report.py fills them from numbers.json and
stops if one is missing. So a number can only reach the PDF through this file, and every one of them can
be traced back: sources.json names the file under results/ that it came from.
"""
if __package__ in (None, ""):  # allow `python src/report_assets.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

from src import analysis
from src.config import RESULTS_DIR, STUDIES, default_config, run_dir, runs_of
from src.data import DATA_DIR, ROOT

GENERATED = ROOT / "report" / "generated"
TABLES = GENERATED / "tables"

SLUGS = {"Otsu (E0)": "otsu", "Tuned VH threshold": "tuned", "Default": "default", "VH only": "vh", "VV+VH": "vvvh",
         "BCE loss": "bce", "Dice loss": "dice", "Scratch U-Net": "scratch", "MobileNetV2 + ImageNet": "imagenet"}
METRICS = {"iou": "IoU", "precision": "precision", "recall": "recall", "f1": "F1", "miou": "mean per-chip IoU"}
MINUS = "−"  # a real minus sign for negative numbers in running text
NBSP = "\u00a0"  # keeps short labels such as "per chip" on one line in narrow table columns


class Numbers:
    """Collects formatted values together with where each one came from."""

    def __init__(self):
        self.values, self.sources = {}, {}

    def add(self, key, value, source):
        if key in self.values:
            raise KeyError(f"number {key!r} defined twice")
        self.values[key] = value
        self.sources[key] = source

    def save(self):
        GENERATED.mkdir(parents=True, exist_ok=True)
        (GENERATED / "numbers.json").write_text(json.dumps(self.values, indent=2, ensure_ascii=False) + "\n")
        (GENERATED / "sources.json").write_text(json.dumps(self.sources, indent=2, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def fmt(value, digits=3):
    return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}f}".replace("-", MINUS)


def pct(value, digits=1):
    return "n/a" if value is None or pd.isna(value) else f"{100 * value:.{digits}f}%"


def thousands(value):
    return f"{int(value):,}"


# ---- numbers --------------------------------------------------------------------------------------------------

def data_numbers(n):
    integrity = read_json(RESULTS_DIR / "data_integrity.json")
    summary = read_json(RESULTS_DIR / "data_summary.json")
    norm = read_json(RESULTS_DIR / "norm_stats.json")
    splits = pd.read_csv(RESULTS_DIR / "split_summary.csv").set_index("split")
    hist = pd.read_csv(RESULTS_DIR / "db_histograms.csv")

    n.add("n_chips_total", str(integrity["chips_in_all_splits"]), "results/data_integrity.json: chips_in_all_splits")
    for split, key in (("train", "n_train"), ("val", "n_val"), ("test", "n_test"), ("bolivia", "n_bolivia")):
        n.add(key, str(integrity["chips_per_split"][split]), f"results/data_integrity.json: chips_per_split.{split}")
    n.add("train_water_share", pct(summary["water_share_of_valid_px"]), "results/data_summary.json: water_share_of_valid_px")
    n.add("train_nodata_share", pct(summary["label_nodata_share"]), "results/data_summary.json: label_nodata_share")
    n.add("train_nan_share", pct(summary["nan_image_share"]), "results/data_summary.json: nan_image_share")
    n.add("train_chips_no_water", str(summary["chips_no_water"]), "results/data_summary.json: chips_no_water")
    n.add("train_chips_lt1", str(summary["chips_water_below_1pct"]), "results/data_summary.json: chips_water_below_1pct")
    n.add("train_chips_lt5", str(summary["chips_water_below_5pct"]), "results/data_summary.json: chips_water_below_5pct")
    clipped = (summary["n_below_clip"] + summary["n_above_clip"]) / summary["n_finite_values"]
    n.add("clipped_share", pct(clipped, 2), "results/data_summary.json: (n_below_clip + n_above_clip) / n_finite_values")

    def overlap(band):
        h = hist[hist["band"] == band]
        w = h[h["class"] == "water"]["count"].to_numpy(float)
        land = h[h["class"] == "not_water"]["count"].to_numpy(float)
        return float(np.minimum(w / w.sum(), land / land.sum()).sum())

    n.add("overlap_vv", fmt(overlap("VV"), 2), "results/db_histograms.csv: shared area of the normalised class histograms, VV")
    n.add("overlap_vh", fmt(overlap("VH"), 2), "results/db_histograms.csv: shared area of the normalised class histograms, VH")
    vh_water = hist[(hist["band"] == "VH") & (hist["class"] == "water")]
    below = vh_water.loc[np.isneginf(vh_water["bin_left"]), "count"].sum() / vh_water["count"].sum()
    n.add("vh_water_below_clip", pct(below), "results/db_histograms.csv: share of VH water pixels below -35 dB")
    for i, ch in enumerate(("vv", "vh", "ratio")):
        n.add(f"norm_mean_{ch}", fmt(norm["mean"][i], 2), f"results/norm_stats.json: mean[{i}]")
        n.add(f"norm_std_{ch}", fmt(norm["std"][i], 2), f"results/norm_stats.json: std[{i}]")
    for split in ("test", "bolivia"):
        px = splits.loc[split, "n_nan_image_px_with_valid_label"]
        n.add(f"{split}_px_no_sar", thousands(px), f"results/split_summary.csv: {split}, n_nan_image_px_with_valid_label")
        n.add(f"{split}_px_no_sar_share", pct(px / splits.loc[split, "n_valid_px"]),
              f"results/split_summary.csv: {split}, n_nan_image_px_with_valid_label / n_valid_px")


def method_numbers(n):
    cfg = default_config()
    for key in ("learning_rate", "batch_size", "crop_size", "crops_per_chip", "epochs", "early_stopping_patience",
                "reduce_lr_patience", "min_learning_rate", "threshold", "bn_momentum"):
        value = cfg[key]
        text = {0.001: "10⁻³", 1e-05: "10⁻⁵"}.get(value, str(value)) if isinstance(value, float) else str(value)
        n.add(f"cfg_{key}", text, f"src/config_default.json: {key}")

    r00, r00b = read_json(run_dir("R00") / "metrics.json"), read_json(run_dir("R00b") / "metrics.json")
    n.add("otsu_threshold", fmt(r00["threshold_db"], 2), "results/R00/metrics.json: threshold_db")
    n.add("tuned_threshold", fmt(r00b["threshold_db"], 2), "results/R00b/metrics.json: threshold_db")

    from src.otsu_baseline import per_chip_otsu_false_alarm

    n.add("per_chip_otsu_false_alarm", pct(float(np.median(per_chip_otsu_false_alarm("train"))), 0),
          "src.otsu_baseline.per_chip_otsu_false_alarm('train'): median over the training chips without water "
          "(also shown in notebooks/02_otsu_baseline.ipynb)")

    params = read_json(RESULTS_DIR / "model_params.json")
    n.add("params_scratch", fmt(params["scratch"]["total_params"] / 1e6, 2), "results/model_params.json: scratch.total_params")
    n.add("params_mobilenet", fmt(params["mobilenetv2"]["total_params"] / 1e6, 2), "results/model_params.json: mobilenetv2.total_params")
    parts = read_json(RESULTS_DIR / "model_parts.json")  # written by notebooks/03_pipeline_models_losses.ipynb
    n.add("params_mobilenet_decoder", fmt(parts["mobilenetv2"]["decoder_params"] / 1e6, 2),
          "results/model_parts.json: mobilenetv2.decoder_params")
    n.add("params_mobilenet_encoder", fmt(parts["mobilenetv2"]["encoder_params"] / 1e6, 2),
          "results/model_parts.json: mobilenetv2.encoder_params")

    prov = read_json(run_dir("R01") / "config.json")["provenance"]
    n.add("tf_version", prov["tensorflow"], "results/R01/config.json: provenance.tensorflow")
    n.add("keras_version", prov["keras"], "results/R01/config.json: provenance.keras")
    n.add("gpu_name", prov["gpu"], "results/R01/config.json: provenance.gpu")

    bench = pd.read_csv(RESULTS_DIR / "smoke" / "step_benchmark.csv", index_col=0)
    n.add("step_ms_plain", fmt(bench.loc["plain TensorFlow graph", "train_step_ms"], 0), "results/smoke/step_benchmark.csv: plain graph, train_step_ms")
    n.add("step_ms_xla", fmt(bench.loc["XLA-compiled", "train_step_ms"], 0), "results/smoke/step_benchmark.csv: XLA, train_step_ms")
    n.add("xla_speedup", fmt(bench.loc["plain TensorFlow graph", "train_step_ms"] / bench.loc["XLA-compiled", "train_step_ms"], 1),
          "results/smoke/step_benchmark.csv: ratio of the two train_step_ms values")
    a = read_json(RESULTS_DIR / "smoke" / "A_bn_keras_default" / "metrics.json")
    b = read_json(RESULTS_DIR / "smoke" / "B_bn_0.99" / "metrics.json")
    n.add("smoke_val_iou_bn999", fmt(a["val"]["iou"], 2), "results/smoke/A_bn_keras_default/metrics.json: val.iou")
    n.add("smoke_val_iou_bn99", fmt(b["val"]["iou"], 2), "results/smoke/B_bn_0.99/metrics.json: val.iou")


def result_numbers(n):
    """Per-config means and stds over the seeds, per-run scores, costs, bootstrap intervals, error analysis."""
    runs = analysis.runs_table()
    done = runs[runs["Status"] == "Finished"]
    summary = analysis.summary_table(runs).set_index("Config")
    columns = {("val", "iou"): "Val IoU (best)", ("test", "iou"): "Test IoU", ("test", "precision"): "Test precision",
               ("test", "recall"): "Test recall", ("test", "f1"): "Test F1", ("test", "miou"): "Test mean per-chip IoU",
               ("bolivia", "iou"): "Bolivia IoU", ("bolivia", "precision"): "Bolivia precision",
               ("bolivia", "recall"): "Bolivia recall", ("bolivia", "f1"): "Bolivia F1",
               ("bolivia", "miou"): "Bolivia mean per-chip IoU"}
    for config, slug in SLUGS.items():
        ids = "R00" if slug == "otsu" else "R00b" if slug == "tuned" else ", ".join(runs_of(config))
        for (split, metric), col in columns.items():
            mean, std = summary.loc[config, f"{col} mean"], summary.loc[config, f"{col} std"]
            src = f"results/all_runs.csv: {col!r}, runs {ids}"
            n.add(f"{slug}_{split}_{metric}", fmt(mean), src + (" (mean)" if "," in ids else ""))
            if "," in ids:
                n.add(f"{slug}_{split}_{metric}_std", fmt(std), src + " (sample standard deviation)")
                n.add(f"{slug}_{split}_{metric}_pm", analysis.mean_std(mean, std), src + " (mean ± sample standard deviation)")
    n.add("n_unet_runs", str(int((~done["Run ID"].isin(analysis.THRESHOLD_RUNS)).sum())), "results/all_runs.csv: finished U-Net runs")
    unet = done[~done["Run ID"].isin(analysis.THRESHOLD_RUNS)]
    for key, col, digits in (("mean_train_minutes", "Train minutes", 0), ("mean_epochs_trained", "Epochs trained", 0),
                             ("mean_sec_per_epoch", "Sec per epoch", 1)):
        n.add(key, fmt(pd.to_numeric(unet[col]).mean(), digits), f"results/all_runs.csv: mean of {col!r} over the U-Net runs")
    n.add("total_train_hours", fmt(pd.to_numeric(unet["Train minutes"]).sum() / 60, 1),
          "results/all_runs.csv: sum of 'Train minutes' over the U-Net runs")

    # every single run
    per_run = {"val_iou": "Val IoU (best)", "test_iou": "Test IoU", "test_precision": "Test precision",
               "test_recall": "Test recall", "bolivia_iou": "Bolivia IoU", "bolivia_recall": "Bolivia recall",
               "bolivia_precision": "Bolivia precision"}
    for _, r in done.iterrows():
        for key, col in per_run.items():
            n.add(f"{r['Run ID']}_{key}", fmt(r[col]), f"results/all_runs.csv: {col!r}, run {r['Run ID']}")
        if not pd.isna(r["Epochs trained"]):
            n.add(f"{r['Run ID']}_epochs", str(int(r["Epochs trained"])), f"results/all_runs.csv: 'Epochs trained', run {r['Run ID']}")
            n.add(f"{r['Run ID']}_best_epoch", str(int(r["Best epoch"])), f"results/all_runs.csv: 'Best epoch', run {r['Run ID']}")

    # differences between config means, test against Bolivia, scores without the no-radar pixels
    def signed(v, digits=3):
        return f"{v:+.{digits}f}".replace("-", MINUS)

    for slug, config in (("otsu", "Otsu (E0)"), ("tuned", "Tuned VH threshold")):
        for split in ("Test", "Bolivia"):
            gap = summary.loc["Default", f"{split} IoU mean"] - summary.loc[config, f"{split} IoU mean"]
            src = f"results/all_runs.csv: mean {split} IoU of R01-R03 minus {split} IoU of the {config} run"
            n.add(f"default_minus_{slug}_{split.lower()}", signed(gap), src)
            n.add(f"default_minus_{slug}_{split.lower()}_abs", fmt(abs(gap)), src + " (absolute value)")
    for config, slug in SLUGS.items():
        gap = summary.loc[config, "Bolivia IoU mean"] - summary.loc[config, "Test IoU mean"]
        n.add(f"{slug}_bolivia_minus_test", signed(gap), f"results/all_runs.csv: mean Bolivia IoU minus mean Test IoU, {config}")
        obs = done[done["Config"] == config]
        for split in ("Test", "Bolivia"):
            n.add(f"{slug}_{split.lower()}_iou_observed", fmt(pd.to_numeric(obs[f"{split} IoU (observed pixels only)"]).mean()),
                  f"results/all_runs.csv: mean of '{split} IoU (observed pixels only)', {config}")
    return runs, summary


def _slug(text):
    return SLUGS.get(text) or "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def analysis_numbers(n):
    """Numbers from the summary and error-analysis notebooks (files they wrote to results/)."""
    def signed(v, digits=3):
        return f"{v:+.{digits}f}".replace("-", MINUS)

    gaps = pd.read_csv(RESULTS_DIR / "gaps.csv")
    for _, g in gaps.iterrows():
        key = f"{_slug(g['A'])}_vs_{_slug(g['B'])}_{g['Metric'].split()[0].lower()}"
        src = f"results/gaps.csv: {g['A']} vs {g['B']}, {g['Metric']}"
        n.add(f"gap_{key}", signed(g["Gap (A - B)"]), src + ", 'Gap (A - B)'")
        n.add(f"gapabs_{key}", fmt(abs(g["Gap (A - B)"])), src + ", absolute gap")
        n.add(f"gapstd_{key}", fmt(g["Larger std"]), src + ", 'Larger std'")
        n.add(f"gapratio_{key}", fmt(g["Gap / std"], 1), src + ", 'Gap / std'")
        n.add(f"welch_{key}", fmt(g["Welch p"], 3 if g["Welch p"] < 0.01 else 2), src + ", 'Welch p'")

    ci = pd.read_csv(RESULTS_DIR / "bootstrap_intervals.csv").set_index("Run")
    for run_id, r in ci.iterrows():
        for split in ("test", "bolivia"):
            low, high = r[f"{split} 95% interval"].split(" to ")
            n.add(f"ci_{run_id}_{split}", f"{float(low):.2f} to {float(high):.2f}",
                  f"results/bootstrap_intervals.csv: {run_id}, {split} 95% interval")
    diffs = pd.read_csv(RESULTS_DIR / "bootstrap_differences.csv")
    for _, d in diffs.iterrows():
        key = f"{d['U-Net']}_{d['Baseline']}_{d['Split']}"
        src = f"results/bootstrap_differences.csv: {d['U-Net']} minus {d['Baseline']}, {d['Split']}"
        low, high = d["95% interval"].split(" to ")
        n.add(f"diff_{key}", signed(d["IoU difference (U-Net - baseline)"]), src)
        n.add(f"diffci_{key}", f"{signed(float(low), 2)} to {signed(float(high), 2)}", src + ", 95% interval")
        n.add(f"diffwins_{key}", pct(d["Resamples in which the U-Net wins"], 0), src + ", share of resamples won")
    for base in ("R00", "R00b"):
        for split in ("test", "bolivia"):
            sub = diffs[(diffs["Baseline"] == base) & (diffs["Split"] == split)]["Resamples in which the U-Net wins"]
            src = f"results/bootstrap_differences.csv: R01-R03 minus {base}, {split}, share of resamples won"
            n.add(f"diffwins_min_{base}_{split}", pct(sub.min(), 0), src + " (lowest of the three seeds)")
            n.add(f"diffwins_max_{base}_{split}", pct(sub.max(), 0), src + " (highest of the three seeds)")

    best = read_json(RESULTS_DIR / "best_run.json")
    n.add("best_run", best["run_id"], "results/best_run.json: run_id")
    n.add("best_run_config", best["config"], "results/best_run.json: config")
    n.add("best_run_seed", str(best["seed"]), "results/best_run.json: seed")

    cost = pd.read_csv(RESULTS_DIR / "cost_table.csv", index_col=0)
    for config, r in cost.iterrows():
        slug, src = SLUGS[config], f"results/cost_table.csv: {config}"
        n.add(f"cost_{slug}_params", fmt(r["Parameters (millions)"], 2), src + ", parameters")
        n.add(f"cost_{slug}_step_ms", fmt(r["Training step (ms), benchmark"], 0), src + ", training step")
        n.add(f"cost_{slug}_infer_ms", fmt(r["Inference per chip (ms), benchmark"], 1), src + ", inference per chip")
        n.add(f"cost_{slug}_gpu_train_mb", fmt(r["Peak GPU memory, training (MB), benchmark"], 0), src + ", peak GPU memory in training")
        n.add(f"cost_{slug}_gpu_infer_mb", fmt(r["Peak GPU memory, inference (MB), benchmark"], 0), src + ", peak GPU memory in inference")
        n.add(f"cost_{slug}_gpu_infer_mb_round", str(int(np.ceil(r["Peak GPU memory, inference (MB), benchmark"] / 100) * 100)),
              src + ", peak GPU memory in inference, rounded up to the next 100 MB")
        n.add(f"cost_{slug}_epochs", fmt(r["Epochs trained, mean of 3 runs"], 0), src + ", epochs trained")
        n.add(f"cost_{slug}_minutes", fmt(r["Training minutes per run, mean of 3 runs"], 1), src + ", minutes per run")
        n.add(f"cost_{slug}_sec_epoch", fmt(r["Seconds per epoch, mean of 3 runs"], 1), src + ", seconds per epoch")
        n.add(f"cost_{slug}_ram_gb", fmt(r["Peak host RAM (MB), mean of 3 runs"] / 1024, 1), src + ", peak host RAM")

    budget = pd.read_csv(RESULTS_DIR / "error_budget.csv", header=[0, 1], index_col=0)
    names = {"missed water: no radar data": "miss_no_radar", "missed water: not dark in the radar image": "miss_not_dark",
             "missed water: thin or small water body": "miss_thin", "missed water: shoreline": "miss_edge",
             "missed water: dark, inside a larger water body": "miss_interior", "false alarm: shoreline": "alarm_edge",
             "false alarm: dark land": "alarm_dark_land", "false alarm: other": "alarm_other"}
    for (split, method) in budget.columns:
        who = "otsu" if method == "Otsu" else "unet"
        for label, short in names.items():
            n.add(f"err_{split}_{who}_{short}", pct(budget.loc[label, (split, method)], 0),
                  f"results/error_budget.csv: {label}, {split}, {method}")
        missed = budget.loc[[k for k in names if k.startswith("missed")], (split, method)].sum()
        n.add(f"err_{split}_{who}_missed_all", pct(missed, 0), f"results/error_budget.csv: all 'missed water' rows, {split}, {method}")

    fails = pd.read_csv(RESULTS_DIR / "failure_groups.csv")
    n.add("n_failing_test", str(int(fails["Test chips"].sum())), "results/failure_groups.csv: sum of 'Test chips'")
    n.add("n_failing_bolivia", str(int(fails["Bolivia chips"].sum())), "results/failure_groups.csv: sum of 'Bolivia chips'")
    n.add("n_failing", str(int(fails["Test chips"].sum() + fails["Bolivia chips"].sum())), "results/failure_groups.csv: all failing chips")
    little = fails[fails["Failure group"].str.startswith("little")].iloc[0]
    n.add("n_failing_little_water", str(int(little["Test chips"] + little["Bolivia chips"])), "results/failure_groups.csv: little or no water")

    wins = pd.read_csv(RESULTS_DIR / "unet_vs_otsu_chips.csv", index_col=0)
    for split in ("test", "bolivia"):
        n.add(f"chips_won_{split}", str(int(wins.loc[split, "chips where the U-Net beats Otsu"])), f"results/unet_vs_otsu_chips.csv: {split}")
        n.add(f"chips_compared_{split}", str(int(wins.loc[split, "chips compared"])), f"results/unet_vs_otsu_chips.csv: {split}")

    events = pd.read_csv(RESULTS_DIR / "per_event_iou.csv", index_col=0)
    unet_col = [c for c in events.columns if c.startswith("U-Net")][0]
    for event, r in events.iterrows():
        e = event.lower().replace("-", "")
        n.add(f"event_{e}_unet", fmt(r[unet_col], 2), f"results/per_event_iou.csv: {event}, {unet_col}")
        n.add(f"event_{e}_otsu", fmt(r["Otsu"], 2), f"results/per_event_iou.csv: {event}, Otsu")

    bright = pd.read_csv(RESULTS_DIR / "water_brightness.csv", index_col=0)
    for name, r in bright.iterrows():
        k = name.lower()
        n.add(f"water_median_vh_{k}", fmt(r["median VH (dB)"], 1), f"results/water_brightness.csv: {name}, median VH")
        n.add(f"water_darker_otsu_{k}", pct(r[[c for c in bright.columns if "Otsu" in c][0]], 0), f"results/water_brightness.csv: {name}, darker than Otsu threshold")
        n.add(f"water_darker_tuned_{k}", pct(r[[c for c in bright.columns if "tuned" in c][0]], 0), f"results/water_brightness.csv: {name}, darker than tuned threshold")

    enc = pd.read_csv(RESULTS_DIR / "bolivia_errors_by_encoder.csv", index_col=0)
    missed = pd.read_csv(RESULTS_DIR / "bolivia_missed_water.csv", index_col=0)
    for col in enc.columns:
        slug = "otsu" if col.startswith("Otsu") else _slug(col.rsplit(" (", 1)[0]) if col.rsplit(" (", 1)[0] != "MobileNetV2" else "default"
        src = f"results/bolivia_errors_by_encoder.csv: {col}"
        n.add(f"bol_{slug}_run", col.rsplit("(", 1)[1].rstrip(")"), src + ", run id")
        n.add(f"bol_{slug}_wrong_k", fmt(enc.loc["all wrong pixels", col] / 1e3, 0), src + ", all wrong pixels (thousands)")
        n.add(f"bol_{slug}_interior_k", fmt(enc.loc["missed water: dark, inside a larger water body", col] / 1e3, 0), src + ", dark interior water missed (thousands)")
        n.add(f"bol_{slug}_notdark_k", fmt(enc.loc["missed water: not dark in the radar image", col] / 1e3, 0), src + ", water not dark missed (thousands)")
        n.add(f"bol_{slug}_alarms_k", fmt(enc.loc[[i for i in enc.index if i.startswith("false alarm")], col].sum() / 1e3, 0), src + ", all false alarms (thousands)")
        n.add(f"bol_{slug}_iou", fmt(enc.loc["pooled IoU on Bolivia", col]), src + ", pooled IoU")
        if col in missed.index:
            m = missed.loc[col]
            n.add(f"bol_{slug}_recall", fmt(m["recall on Bolivia"], 2), f"results/bolivia_missed_water.csv: {col}, recall")
            n.add(f"bol_{slug}_missed_median_vh", fmt(m["median VH of the water it missed (dB)"], 1), f"results/bolivia_missed_water.csv: {col}, median VH of missed water")
            n.add(f"bol_{slug}_found_median_vh", fmt(m["median VH of the water it found (dB)"], 1), f"results/bolivia_missed_water.csv: {col}, median VH of found water")
            n.add(f"bol_{slug}_missed_dark_share", pct(m["missed water that is darker than the Otsu threshold"], 0), f"results/bolivia_missed_water.csv: {col}, missed water darker than Otsu")

    thr = pd.read_csv(run_dir("R01") / "val_threshold_check.csv", index_col=0)
    n.add("r01_val_iou_thr05", fmt(thr.loc[0.5, "IoU"]), "results/R01/val_threshold_check.csv: IoU at threshold 0.5")
    n.add("r01_val_iou_best_thr", fmt(thr["IoU"].max()), "results/R01/val_threshold_check.csv: highest IoU over the thresholds")
    n.add("r01_best_thr", fmt(float(thr["IoU"].idxmax()), 1), "results/R01/val_threshold_check.csv: threshold with the highest IoU")
    n.add("thr_gain_r01", fmt(thr["IoU"].max() - thr.loc[0.5, "IoU"]), "results/R01/val_threshold_check.csv: highest IoU minus IoU at 0.5")


# ---- tables ---------------------------------------------------------------------------------------------------

def md_table(frame, align=None):
    """A pandas frame as a Markdown pipe table. `align` is a string of l / r, one per column."""
    cols = list(frame.columns)
    align = align or "l" + "r" * (len(cols) - 1)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(":---" if a == "l" else "---:" for a in align) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_table(name, caption, frame, align=None):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.md").write_text(f"Table: {caption} {{#tbl:{name}}}\n\n{md_table(frame, align)}\n")


def table_events():
    """The eleven hand-labelled flood events: dates and orbit from the dataset's metadata, chip counts from the splits."""
    integrity = read_json(RESULTS_DIR / "data_integrity.json")
    meta = {f["properties"]["location"]: f["properties"] for f in read_json(DATA_DIR / "metadata.geojson")["features"]}
    meta["Mekong"] = meta["Cambodia"]  # the chips of the Cambodian event are named after the Mekong
    per_split = pd.DataFrame(integrity["events_per_split"]).fillna(0).astype(int)
    rows = []
    for event in sorted(per_split.index, key=lambda e: (e == "Bolivia", e)):
        m = meta[event]
        rows.append({"Event": event, "Radar image": m["s1_date"].replace("/", "-"), "Orbit": m["orbit"].lower(),
                     "Hand-labelled chips": int(per_split.loc[event].sum()), "Train": per_split.loc[event, "train"],
                     "Validation": per_split.loc[event, "val"], "Test": per_split.loc[event, "test"],
                     "Held out": per_split.loc[event, "bolivia"]})
    frame = pd.DataFrame(rows)
    total = {"Event": "**Total**", "Radar image": "", "Orbit": "", **{c: f"**{frame[c].sum()}**" for c in frame.columns[3:]}}
    frame = pd.concat([frame, pd.DataFrame([total])], ignore_index=True).replace(0, "–")
    write_table("events", "The eleven flood events of the hand-labelled subset and the official split. "
                "Dates and orbit direction are taken from the dataset's metadata file.", frame, "lllrrrrr")


def table_splits():
    splits = pd.read_csv(RESULTS_DIR / "split_summary.csv")
    names = {"train": "Train", "val": "Validation", "test": "Test", "bolivia": "Bolivia (held out)"}
    frame = pd.DataFrame({
        "Split": splits["split"].map(names),
        "Chips": splits["n_chips"],
        "Events": splits["n_events"],
        "Labelled pixels (millions)": (splits["n_valid_px"] / 1e6).map("{:.1f}".format),
        "Water share of labelled pixels": splits["water_share_of_valid_px"].map(pct),
        "Pixels without label": splits["label_nodata_share"].map(pct),
        "Chips without water": splits["chips_no_water"],
        "Labelled pixels without radar value": splits["n_nan_image_px_with_valid_label"].map(thousands),
    })
    write_table("splits", "The four splits. Only the training column was used for design decisions; the other "
                "columns are label bookkeeping.", frame, "lrrrrrrr")


def _pm(summary, config, column):
    return analysis.mean_std(summary.loc[config, f"{column} mean"], summary.loc[config, f"{column} std"])


def table_main(summary):
    """Threshold rules against the default U-Net, test and Bolivia."""
    rows = []
    for split in ("Test", "Bolivia"):
        for config, name in (("Otsu (E0)", "Otsu threshold (R00)"), ("Tuned VH threshold", "Tuned threshold (R00b)"),
                             ("Default", "Default U-Net (R01–R03)")):
            rows.append({"Split": split if config == "Otsu (E0)" else "", "Method": name,
                         "Water IoU": _pm(summary, config, f"{split} IoU"),
                         "Precision": _pm(summary, config, f"{split} precision"),
                         "Recall": _pm(summary, config, f"{split} recall"),
                         "F1": _pm(summary, config, f"{split} F1"),
                         "Mean per-chip IoU": _pm(summary, config, f"{split} mean per-chip IoU")})
    write_table("main", "The default U-Net against the two threshold rules. U-Net values are the mean ± standard "
                "deviation of three seeds; the threshold rules are deterministic. IoU, precision, recall and F1 are "
                "pooled over all labelled pixels of the split.", pd.DataFrame(rows), "llrrrrr")


def table_ablations(summary):
    rows = []
    layout = (("Reference", ("Default",)), ("E1 · input", ("VH only", "VV+VH")), ("E2 · loss", ("BCE loss", "Dice loss")),
              ("E3 · encoder", ("Scratch U-Net", "MobileNetV2 + ImageNet")))
    for study, configs in layout:
        for i, config in enumerate(configs):
            rows.append({"Study": study if i == 0 else "", "Configuration": "Default (VV+VH+ratio, BCE+Dice, MobileNetV2)"
                         if config == "Default" else config,
                         "Validation IoU": _pm(summary, config, "Val IoU (best)"), "Test IoU": _pm(summary, config, "Test IoU"),
                         "Test precision": _pm(summary, config, "Test precision"), "Test recall": _pm(summary, config, "Test recall"),
                         "Bolivia IoU": _pm(summary, config, "Bolivia IoU")})
    write_table("ablations", "The three ablation studies: mean ± standard deviation over three seeds. Each "
                "configuration differs from the default in exactly one factor.", pd.DataFrame(rows), "llrrrrr")


def table_gaps():
    path = RESULTS_DIR / "gaps.csv"
    if not path.exists():
        return
    gaps = pd.read_csv(path)
    rows = []
    for (study, a, b), g in gaps.groupby(["Study", "A", "B"], sort=False):
        row = {"Study": study, "Comparison": f"{a} − {b}"}
        for metric, name in (("Test IoU", "Test"), ("Bolivia IoU", "Bolivia")):
            m = g[g["Metric"] == metric].iloc[0]
            row[f"{name}: gap"] = f"{m['Gap (A - B)']:+.3f}".replace("-", MINUS)
            row[f"{name}: larger std"] = f"{m['Larger std']:.3f}"
            row[f"{name}: real?"] = "yes" if m["Gap larger than std"] else "no"
        rows.append(row)
    write_table("gaps", "Every pair of configurations within a study: difference of the seed means and the larger of "
                "the two seed-to-seed standard deviations. A gap counts as real only if it exceeds that deviation.",
                pd.DataFrame(rows), "llrrlrrl")


def table_cost(summary):
    path = RESULTS_DIR / "cost_table.csv"
    if not path.exists():
        return
    cost = pd.read_csv(path, index_col=0)
    frame = pd.DataFrame({
        "Encoder": ["Scratch U-Net", "MobileNetV2 (default)", "MobileNetV2 + ImageNet"],
        "Parameters (millions)": cost["Parameters (millions)"].map("{:.2f}".format).values,
        "Training step (ms)": cost["Training step (ms), benchmark"].map("{:.0f}".format).values,
        "Prediction per chip (ms)": cost["Inference per chip (ms), benchmark"].map("{:.1f}".format).values,
        "Peak GPU memory, training (MB)": cost["Peak GPU memory, training (MB), benchmark"].map("{:.0f}".format).values,
        "Epochs until stop": cost["Epochs trained, mean of 3 runs"].map("{:.0f}".format).values,
        "Minutes per run": cost["Training minutes per run, mean of 3 runs"].map("{:.1f}".format).values,
        "Test IoU": [_pm(summary, c, "Test IoU") for c in cost.index],
    })
    write_table("cost", "What each encoder costs and what it delivers. Step time, prediction time and memory come from "
                "a controlled benchmark of each model alone on the GPU (batch of 8 crops of 256×256 pixels; one "
                "512×512 chip); epochs and minutes are means of the three training runs.", frame, "lrrrrrrr")


def table_published(summary):
    """Published numbers (report/published_results.csv, each read in the paper named there) next to this work."""
    pub = pd.read_csv(ROOT / "report" / "published_results.csv", dtype=str).fillna("")  # keep the reported precision
    rows = []
    for _, r in pub.iterrows():
        source = f"[@{r['citation']}]" + (f", as listed in [@{r['listed_in']}]" if r["listed_in"] else "")
        rows.append({"Method": f"{r['method']} {source}", "Input": r["input"], "IoU": r["iou_kind"].replace(" ", NBSP),
                     "Test": r["test_iou"] or "–", "Bolivia": r["bolivia_iou"] or "–"})
    millions = read_json(RESULTS_DIR / "model_params.json")["mobilenetv2"]["total_params"] / 1e6
    ours = (("Otsu threshold on VH", "Otsu (E0)", "Sentinel-1: VH"), ("Tuned VH threshold", "Tuned VH threshold", "Sentinel-1: VH"),
            (f"**Default U-Net (MobileNetV2 encoder, {millions:.1f} M parameters)**", "Default", "Sentinel-1: VV, VH, ratio"))
    for kind, column in (("pooled", "IoU"), ("per chip", "mean per-chip IoU")):
        for name, config, inputs in ours:
            rows.append({"Method": f"This work: {name}", "Input": inputs, "IoU": kind.replace(" ", NBSP),
                         "Test": _pm(summary, config, f"Test {column}"), "Bolivia": _pm(summary, config, f"Bolivia {column}")})
    write_table("published", "Published water IoU on the hand-labelled test chips of Sen1Floods11 and on Bolivia, next to "
                "this work. \"Pooled\": one IoU over all pixels of the split; \"per chip\": the mean of the chips' IoU "
                "values. Training data, inputs and model selection differ between papers (see text), so the comparison "
                "is indicative only.", pd.DataFrame(rows), "lllrr")


def table_all_runs(runs):
    done = runs[runs["Status"] == "Finished"]
    frame = pd.DataFrame({
        "Run": done["Run ID"], "Configuration": done["Config"], "Seed": done["Seed"],
        "Epochs": done["Epochs trained"].map(lambda v: "–" if pd.isna(v) else f"{int(v)}"),
        "Best epoch": done["Best epoch"].map(lambda v: "–" if pd.isna(v) else f"{int(v)}"),
        "Val IoU": done["Val IoU (best)"].map(fmt), "Test IoU": done["Test IoU"].map(fmt),
        "Test precision": done["Test precision"].map(fmt), "Test recall": done["Test recall"].map(fmt),
        "Bolivia IoU": done["Bolivia IoU"].map(fmt),
        "s / epoch": done["Sec per epoch"].map(lambda v: "–" if pd.isna(v) else f"{v:.1f}"),
    })
    write_table("allruns", "Every run. The same table, with more columns, is results/all_runs.csv.", frame, "lllrrrrrrrr")


def table_failures():
    path = RESULTS_DIR / "failure_groups.csv"
    if not path.exists():
        return
    f = pd.read_csv(path)
    frame = pd.DataFrame({
        "Failure group": f["Failure group"], "Test chips": f["Test chips"], "Bolivia chips": f["Bolivia chips"],
        "Mean IoU, U-Net": f["Mean IoU"].map(fmt), "Mean IoU, Otsu, same chips": f["Otsu mean IoU, same chips"].map(fmt),
        "Example chip": f["Example chip"].str.replace("_", " "), "Its IoU": f["Example IoU"].map(fmt),
    })
    write_table("failures", "Chips on which the best U-Net fails (IoU below 0.5), grouped by a fixed rule: under 1% "
                "water, otherwise the category holding most of the chip's wrong pixels.", frame, "lrrrrlr")


def table_errors():
    path = RESULTS_DIR / "error_budget.csv"
    if not path.exists():
        return
    b = pd.read_csv(path, header=[0, 1], index_col=0)
    frame = pd.DataFrame({"Kind of error": b.index})
    for (split, method) in b.columns:
        frame[f"{'Test' if split == 'test' else 'Bolivia'}: {method}"] = b[(split, method)].map(lambda v: pct(v, 0)).values
    write_table("errors", "Where the wrong pixels are: share of all wrong pixels of a method on a split, by "
                "category. \"Dark\" means VH below the Otsu threshold.", frame, "lrrrr")


def write_readme_block(summary):
    """Refresh the results table between the two markers in README.md."""
    readme = ROOT / "README.md"
    text = readme.read_text()
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    if start not in text or end not in text:
        return
    rows = [("Otsu threshold on VH (R00)", "Otsu (E0)"), ("VH threshold tuned on the training labels (R00b)", "Tuned VH threshold"),
            ("**Default U-Net** (MobileNetV2, VV+VH+ratio, BCE+Dice; R01–R03)", "Default"),
            ("U-Net, VH only (R04–R06)", "VH only"), ("U-Net, VV+VH (R07–R09)", "VV+VH"),
            ("U-Net, BCE loss (R10–R12)", "BCE loss"), ("U-Net, Dice loss (R13–R15)", "Dice loss"),
            ("Scratch U-Net (R16–R18)", "Scratch U-Net"), ("MobileNetV2 + ImageNet weights (R19–R21)", "MobileNetV2 + ImageNet")]
    lines = ["## Results at a glance", "",
             "Pooled water IoU, mean ± standard deviation over three seeds (generated from `results/` by "
             "`python -m src.report_assets`):", "",
             "| Method | Validation IoU | Test IoU | Bolivia IoU (unseen flood) |", "|:---|---:|---:|---:|"]
    for name, config in rows:
        if summary.loc[config, "Finished runs"] > 0:
            lines.append(f"| {name} | {_pm(summary, config, 'Val IoU (best)')} | {_pm(summary, config, 'Test IoU')} "
                         f"| {_pm(summary, config, 'Bolivia IoU')} |")
    block = "\n".join(lines)
    readme.write_text(text[:text.index(start) + len(start)] + "\n" + block + "\n" + text[text.index(end):])


def main():
    n = Numbers()
    data_numbers(n)
    method_numbers(n)
    runs, summary = result_numbers(n)
    analysis_numbers(n)
    table_events()
    table_splits()
    table_main(summary)
    table_ablations(summary)
    table_gaps()
    table_cost(summary)
    table_published(summary)
    table_all_runs(runs)
    table_failures()
    table_errors()
    write_readme_block(summary)
    n.save()
    print(f"wrote {len(n.values)} numbers and {len(list(TABLES.glob('*.md')))} tables to {GENERATED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
