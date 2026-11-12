"""Inference demo: map the water in one Sentinel-1 chip with a trained model.

    python src/predict.py --chip data/s1/Bolivia_103757_S1Hand.tif
    python src/predict.py --chip <path> --run R01 --out my_chip.png --tif my_mask.tif

What it does, end to end and outside the training code:
  1. loads the best-validation model of a run (default: the run named in results/best_run.json,
     i.e. the default-config model with the highest validation IoU; R01 if that file is missing);
  2. reads the chip (a 2-band GeoTIFF: VV, VH in dB) and preprocesses it with THE SAME function and
     the same results/norm_stats.json as training (src/preprocess.py);
  3. predicts the water probability of every pixel and thresholds it at the run's threshold (0.5);
  4. saves a picture (VH next to the predicted mask; if a hand label exists, also the label and an
     error map) and, on request, the mask as a georeferenced GeoTIFF that opens in QGIS.
"""
if __package__ in (None, ""):  # allow `python src/predict.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.config import RESULTS_DIR, run_dir
from src.data import LABEL_DIR, NO_DATA, ROOT, WATER, chip_name, read_image, read_label
from src.metrics import chip_scores
from src.preprocess import finite_mask, preprocess

BEST_RUN_JSON = RESULTS_DIR / "best_run.json"
DEMO_DIR = ROOT / "figures" / "demo"


def default_run():
    """The run the demo uses unless told otherwise."""
    if BEST_RUN_JSON.exists():
        return json.loads(BEST_RUN_JSON.read_text())["run_id"]
    return "R01"


def load_model(run_id):
    """(model, config) of a finished run. The model is inference-ready and needs no custom objects."""
    import keras

    from src.train import setup_runtime

    cfg = json.loads((run_dir(run_id) / "config.json").read_text())
    setup_runtime(cfg["mixed_precision"])
    model = keras.models.load_model(run_dir(run_id) / "best.keras", compile=False)
    model.jit_compile = False  # single images: the plain graph starts instantly, no compile wait
    return model, cfg


def predict_proba(model, cfg, image):
    """Raw chip (2, H, W) [VV, VH] in dB -> water probability (H, W) float32.

    Any size works: the input is padded with zeros (the average pixel after standardising) to the next
    multiple of 32, which the network needs, and the padding is cut off again.
    """
    x = preprocess(image, cfg["channels"])  # the training-time preprocessing, nothing else
    h, w = x.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)))
    prob = np.asarray(model.predict_on_batch(x[None]), dtype=np.float32)[0, ..., 0]
    return prob[:h, :w]


def find_label(chip_path):
    """The hand label that belongs to a Sen1Floods11 chip, if it is on disk."""
    path = LABEL_DIR / f"{chip_name(chip_path)}_LabelHand.tif"
    return path if path.exists() else None


def save_figure(image, mask, label, title, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import viz

    viz.use_style()
    n = 4 if label is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.75))
    viz.show_db(axes[0], image[1], "VH", title="VH backscatter (dB)")
    viz.show_label(axes[1], np.where(finite_mask(image), mask, NO_DATA), title="predicted water")
    handles = viz.label_legend_handles()
    if label is not None:
        viz.show_label(axes[2], label, title="hand label")
        viz.show_errors(axes[3], mask, label, title="prediction vs label")
        handles = viz.error_legend_handles()
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.97), w_pad=0.5)
    viz.savefig(fig, Path(out_path))
    plt.close(fig)


def save_geotiff(mask, image, chip_path, out_path):
    """Write the mask (1 water, 0 not water, 255 where the chip has no radar data) with the chip's georeferencing."""
    import rasterio

    out = np.where(finite_mask(image), mask.astype(np.uint8), np.uint8(255))
    with rasterio.open(chip_path) as src:
        profile = src.profile
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "water mask: 1 = water, 0 = not water")


def predict_file(chip_path, run_id=None, out_path=None, tif_path=None, log=print):
    """Run the whole demo for one chip. Returns a small summary dict."""
    chip_path = Path(chip_path)
    run_id = run_id or default_run()
    model, cfg = load_model(run_id)

    image = read_image(chip_path)
    t0 = time.perf_counter()
    prob = predict_proba(model, cfg, image)
    seconds = time.perf_counter() - t0
    mask = (prob >= cfg["threshold"]).astype(np.int8)

    observed = finite_mask(image)
    summary = {"chip": chip_path.name, "run": run_id, "size": list(mask.shape),
               "water_share_of_observed_pixels": float(mask[observed].mean()) if observed.any() else float("nan"),
               "pixels_without_radar_data": int((~observed).sum()), "seconds": round(seconds, 2)}

    label_path = find_label(chip_path)
    label = read_label(label_path) if label_path else None
    title = f"{chip_name(chip_path).replace('_', ' ')} · model {run_id} ({cfg.get('config_name', '')})"
    if label is not None and (label != NO_DATA).any():
        scores = chip_scores(mask, label)
        summary.update({"iou": scores["iou"], "precision": scores["precision"], "recall": scores["recall"],
                        "label_water_share": float((label == WATER).sum() / (label != NO_DATA).sum())})
        if not np.isnan(scores["iou"]):
            title += f" · IoU {scores['iou']:.2f}"

    out_path = Path(out_path) if out_path else DEMO_DIR / f"{chip_name(chip_path)}_{run_id}.png"
    save_figure(image, mask, label, title, out_path)
    summary["figure"] = str(out_path)
    if tif_path:
        save_geotiff(mask, image, chip_path, tif_path)
        summary["geotiff"] = str(tif_path)

    log(f"chip   : {chip_path.name}  ({mask.shape[0]}x{mask.shape[1]} pixels)")
    log(f"model  : {run_id}, {cfg.get('config_name', '')} ({cfg['encoder']}, channels {cfg['channels']}, loss {cfg['loss']})")
    log(f"water  : {summary['water_share_of_observed_pixels']:.1%} of the pixels with radar data ({seconds:.2f} s)")
    if "iou" in summary:
        log(f"label  : {summary['label_water_share']:.1%} water | IoU {summary['iou']:.3f}  "
            f"precision {summary['precision']:.3f}  recall {summary['recall']:.3f}")
    log(f"figure : {out_path}")
    if tif_path:
        log(f"geotiff: {tif_path}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Predict the water mask of one Sentinel-1 chip.")
    p.add_argument("--chip", required=True, help="path of a 2-band GeoTIFF (VV, VH in dB), e.g. data/s1/Bolivia_103757_S1Hand.tif")
    p.add_argument("--run", help="run id of the model to use (default: results/best_run.json, else R01)")
    p.add_argument("--out", help="where to save the PNG (default: figures/demo/<chip>_<run>.png)")
    p.add_argument("--tif", help="also save the mask as a georeferenced GeoTIFF at this path")
    args = p.parse_args()
    predict_file(args.chip, args.run, args.out, args.tif)


if __name__ == "__main__":
    main()
