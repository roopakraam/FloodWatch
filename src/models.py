"""U-Net models with three encoder options.

    build_unet("scratch", in_channels)              5 levels, 16-32-64-128-256 filters, trained from nothing
    build_unet("mobilenetv2", in_channels)          MobileNetV2 encoder, random initial weights
    build_unet("mobilenetv2_imagenet", in_channels) MobileNetV2 encoder, ImageNet weights (needs 3 channels)

All models take (H, W, in_channels) with H and W divisible by 32 (the spatial size is left open, so the
same model trains on 256 crops and is evaluated on full 512 chips) and return a (H, W, 1) sigmoid map
= probability of water.

`bn_momentum` is applied to EVERY BatchNormalization layer, including the ones inside MobileNetV2.
Keras' MobileNetV2 ships with momentum 0.999, which needs thousands of updates before the moving
statistics used at inference catch up with the weights. A run here has only 126 updates per epoch, so
with 0.999 the validation score lags many epochs behind training (see notebooks/04_smoke_test.ipynb).

Run from the project root to build all three, check output shapes and save parameter counts:
    python -m src.models
"""
import json

import keras
from keras import layers

from src.config import ENCODER_LABELS  # noqa: F401  (re-exported for notebooks)
from src.data import ROOT

ENCODERS = ("scratch", "mobilenetv2", "mobilenetv2_imagenet")

SCRATCH_FILTERS = (16, 32, 64, 128, 256)

# MobileNetV2 feature maps used as skip connections, from H/2 down to H/32 (5 sizes).
# The last one is the bottleneck that the decoder starts from.
MOBILENET_SKIPS = (
    "block_1_expand_relu",    # H/2,   96 channels
    "block_3_expand_relu",    # H/4,  144
    "block_6_expand_relu",    # H/8,  192
    "block_13_expand_relu",   # H/16, 576
    "block_16_project",       # H/32, 320
)
MOBILENET_DECODER_FILTERS = (256, 128, 64, 32)
BN_MOMENTUM = 0.99  # Keras' default for a BatchNormalization layer


def conv_block(x, filters, name):
    """Two 3x3 conv + BatchNorm + ReLU."""
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.ReLU(name=f"{name}_relu{i}")(x)
    return x


def up_block(x, skip, filters, name):
    """Upsample x by 2 (nearest neighbour), concatenate the skip connection, then conv_block."""
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
    encoder = keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet" if imagenet else None,
    )
    feats = [encoder.get_layer(n).output for n in MOBILENET_SKIPS]
    x, skips = feats[-1], feats[-2::-1]  # bottleneck, then skips from deep to shallow
    for level, (filters, skip) in enumerate(zip(MOBILENET_DECODER_FILTERS, skips), start=1):
        x = up_block(x, skip, filters, f"dec{level}")
    # x is at H/2 now: one more upsample back to full size (MobileNetV2 has no full-resolution feature map)
    x = layers.UpSampling2D(2, interpolation="nearest", name="dec5_up")(x)
    return conv_block(x, 16, "dec5")


def build_unet(encoder, in_channels, bn_momentum=BN_MOMENTUM):
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

    # float32 output even under mixed precision, so the sigmoid and the loss stay numerically stable
    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="water_prob")(x)
    model = keras.Model(inputs, outputs, name=f"unet_{encoder}")

    if bn_momentum is not None:
        for layer in model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.momentum = bn_momentum
    return model


def count_parameters(model):
    trainable = int(sum(w.numpy().size for w in model.trainable_weights))
    return {"total_params": int(model.count_params()), "trainable_params": trainable}


def main():
    import numpy as np

    counts = {}
    for enc in ENCODERS:
        model = build_unet(enc, 3)
        for size in (256, 512):
            out = model(np.zeros((1, size, size, 3), dtype="float32"))
            assert out.shape == (1, size, size, 1), out.shape
        counts[enc] = count_parameters(model)
        print(f"{enc:22s} {counts[enc]['total_params']:>10,} parameters "
              f"({counts[enc]['trainable_params']:,} trainable) | output OK for 256 and 512 input")

    out_path = ROOT / "results" / "model_params.json"
    out_path.write_text(json.dumps(counts, indent=2) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
