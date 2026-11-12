"""Controlled timing of one model on fixed synthetic batches: training step, inference, GPU memory.

The seconds-per-epoch logged by a real run depend on what else the laptop was doing and how warm it
was. For a fair cost comparison between encoders this script times each model the same way, alone on
the GPU: N training steps on one batch of 8 crops (256x256) and N inference calls on one batch of 8
full chips (512x512), after a warm-up, reporting medians.

Run from the project root (prints one JSON object as the last line):
    python -m src.benchmark --encoder mobilenetv2
    python -m src.benchmark --encoder mobilenetv2 --no-jit        # without XLA, for the smoke test
"""
if __package__ in (None, ""):  # allow `python src/benchmark.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import numpy as np

from src.config import default_config
from src.models import ENCODERS
from src.preprocess import CHANNEL_SETS, n_channels


def benchmark(encoder="mobilenetv2", channels="vv_vh_ratio", loss="bce_dice", jit_compile=True, steps=30):
    import keras
    import tensorflow as tf

    from src.losses import LOSSES, WaterIoU
    from src.models import build_unet, count_parameters
    from src.train import setup_runtime

    cfg = default_config()
    gpus = setup_runtime(cfg["mixed_precision"])
    keras.utils.set_random_seed(0)
    c, batch, crop = n_channels(channels), cfg["batch_size"], cfg["crop_size"]
    model = build_unet(encoder, c, bn_momentum=cfg["bn_momentum"])
    model.compile(optimizer=keras.optimizers.Adam(cfg["learning_rate"]), loss=LOSSES[loss],
                  metrics=[WaterIoU()], jit_compile=jit_compile)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(batch, crop, crop, c)).astype("float32")
    y = rng.integers(-1, 2, size=(batch, crop, crop, 1)).astype("float32")
    x_full = rng.normal(size=(cfg["eval_batch_size"], 512, 512, c)).astype("float32")

    def timed(fn, n):
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t0)
        return out

    def peak_mb():
        return round(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1024**2, 1) if gpus else None

    if gpus:
        tf.config.experimental.reset_memory_stats("GPU:0")
    first_train = timed(lambda: model.train_on_batch(x, y), 1)[0]
    timed(lambda: model.train_on_batch(x, y), 5)                      # warm-up
    train = timed(lambda: model.train_on_batch(x, y), steps)
    train_peak = peak_mb()

    if gpus:
        tf.config.experimental.reset_memory_stats("GPU:0")
    first_infer = timed(lambda: model.predict_on_batch(x_full), 1)[0]
    timed(lambda: model.predict_on_batch(x_full), 3)
    infer = timed(lambda: model.predict_on_batch(x_full), steps)
    infer_peak = peak_mb()

    return {
        "encoder": encoder, "channels": channels, "loss": loss, "jit_compile": jit_compile,
        "mixed_precision": cfg["mixed_precision"], "batch_size": batch, "crop_size": crop,
        **count_parameters(model),
        "first_train_step_s": round(first_train, 1),
        "train_step_ms": round(1000 * float(np.median(train)), 1),
        "train_epoch_s_126_steps": round(126 * float(np.median(train)), 1),
        "peak_gpu_memory_train_mb": train_peak,
        "first_inference_s": round(first_infer, 1),
        "inference_ms_per_chip": round(1000 * float(np.median(infer)) / len(x_full), 1),
        "peak_gpu_memory_inference_mb": infer_peak,
        "gpu": tf.config.experimental.get_device_details(gpus[0]).get("device_name") if gpus else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--encoder", default="mobilenetv2", choices=ENCODERS)
    p.add_argument("--channels", default="vv_vh_ratio", choices=list(CHANNEL_SETS))
    p.add_argument("--loss", default="bce_dice")
    p.add_argument("--no-jit", action="store_true", help="plain TensorFlow graph instead of XLA")
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()
    result = benchmark(args.encoder, args.channels, args.loss, not args.no_jit, args.steps)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
