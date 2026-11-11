"""U-Net models with three encoder options.

    build_unet("scratch", in_channels)              5 levels, 16-32-64-128-256 filters, trained from nothing
    build_unet("mobilenetv2", in_channels)          MobileNetV2 encoder, random weights
    build_unet("mobilenetv2_imagenet", in_channels) MobileNetV2 encoder, ImageNet weights (needs 3 channels)

All models take (H, W, in_channels) with H and W divisible by 32 (spatial size is left open so
256 crops for training and 512 chips for evaluation use the same model) and return a
(H, W, 1) sigmoid map = probability of water.

Run from the project root to build all three, check output shapes and save parameter counts:
    python -m src.models
"""
import json

import tensorflow as tf
from tensorflow.keras import layers

from src.data import ROOT

ENCODERS = ("scratch", "mobilenetv2", "mobilenetv2_imagenet")

SCRATCH_FILTERS = (16, 32, 64, 128, 256)

# MobileNetV2 feature maps used as skip connections, from H/2 down to H/32 (5 sizes).
# The last one is the bottleneck that the decoder starts from.
MOBILENET_SKIPS = (
    "block_1_expand_relu",    # H/2
    "block_3_expand_relu",    # H/4
    "block_6_expand_relu",    # H/8
    "block_13_expand_relu",   # H/16
    "block_16_project",       # H/32
)
MOBILENET_DECODER_FILTERS = (256, 128, 64, 32)


def conv_block(x, filters, name):
    """Two 3x3 conv + BatchNorm + ReLU."""
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.ReLU(name=f"{name}_relu{i}")(x)
    return x


def up_block(x, skip, filters, name):
    """Upsample x by 2 (nearest: bilinear gives float32 gradients that break mixed precision), concatenate the skip connection, then conv_block."""
    x = layers.UpSampling2D(2, interpolation="nearest", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    return conv_block(x, filters, name)


def _scratch_unet(inputs):
    skips = []
    x = inputs
    for level, filters in enumerate(SCRATCH_FILTERS, start=1):
        x = conv_block(x, filters, f"enc{level}")
        if level < len(SCRATCH_FILTERS):  # the 5th level is the bottleneck
            skips.append(x)
            x = layers.MaxPooling2D(2, name=f"enc{level}_pool")(x)
    for level, (filters, skip) in enumerate(zip(SCRATCH_FILTERS[-2::-1], skips[::-1]), start=1):
        x = up_block(x, skip, filters, f"dec{level}")
    return x


def _mobilenet_unet(inputs, imagenet):
    encoder = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet" if imagenet else None,
    )
    feats = [encoder.get_layer(n).output for n in MOBILENET_SKIPS]
    x, skips = feats[-1], feats[-2::-1]  # bottleneck, then skips from deep to shallow
    for level, (filters, skip) in enumerate(zip(MOBILENET_DECODER_FILTERS, skips), start=1):
        x = up_block(x, skip, filters, f"dec{level}")
    # x is at H/2 now: one more upsample back to full size (no skip at full resolution)
    x = layers.UpSampling2D(2, interpolation="nearest", name="dec5_up")(x)
    return conv_block(x, 16, "dec5")


def build_unet(encoder, in_channels):
    """Build a U-Net. `encoder` is one of ENCODERS. Returns a Keras model."""
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
    if encoder == "mobilenetv2_imagenet" and in_channels != 3:
        raise ValueError("ImageNet weights need exactly 3 input channels (use VV+VH+ratio)")

    inputs = layers.Input((None, None, in_channels), name="image")
    if encoder == "scratch":
        x = _scratch_unet(inputs)
    else:
        x = _mobilenet_unet(inputs, imagenet=(encoder == "mobilenetv2_imagenet"))

    # float32 output even under mixed precision, so the sigmoid/loss stay stable
    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="water_prob")(x)
    return tf.keras.Model(inputs, outputs, name=f"unet_{encoder}")


def main():
    import numpy as np

    counts = {}
    for enc in ENCODERS:
        model = build_unet(enc, 3)
        model.summary(line_length=100, show_trainable=False)
        for size in (256, 512):
            out = model(np.zeros((1, size, size, 3), dtype="float32"))
            assert out.shape == (1, size, size, 1), out.shape
        counts[enc] = {
            "total_params": model.count_params(),
            "trainable_params": int(sum(tf.size(w) for w in model.trainable_weights)),
        }
        print(f"== {enc}: {counts[enc]['total_params']:,} parameters, output OK for 256 and 512 input\n")

    out_path = ROOT / "results" / "model_params.json"
    out_path.write_text(json.dumps(counts, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
