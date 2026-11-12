"""Tests for src/models.py: every encoder builds, keeps the spatial size and uses the BN momentum."""
import json

import keras
import numpy as np
import pytest

from src.data import ROOT
from src.models import ENCODERS, build_unet, count_parameters


@pytest.mark.parametrize("encoder", ENCODERS)
def test_output_shape_matches_input_for_256_and_512(encoder):
    model = build_unet(encoder, 3)
    for size in (256, 512):
        out = model(np.zeros((1, size, size, 3), dtype="float32"))
        assert out.shape == (1, size, size, 1)
    out = np.asarray(out)
    assert out.dtype == np.float32 and out.min() >= 0.0 and out.max() <= 1.0  # sigmoid probabilities


@pytest.mark.parametrize("encoder", ["scratch", "mobilenetv2"])
@pytest.mark.parametrize("channels", [1, 2, 3])
def test_one_two_or_three_input_channels(encoder, channels):
    model = build_unet(encoder, channels)
    assert model(np.zeros((2, 64, 64, channels), dtype="float32")).shape == (2, 64, 64, 1)


def test_imagenet_weights_need_three_channels():
    with pytest.raises(ValueError):
        build_unet("mobilenetv2_imagenet", 2)


def test_unknown_encoder_is_rejected():
    with pytest.raises(ValueError):
        build_unet("resnet50", 3)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_bn_momentum_reaches_every_batchnorm_layer(encoder):
    model = build_unet(encoder, 3, bn_momentum=0.9)
    bns = [l for l in model.layers if isinstance(l, keras.layers.BatchNormalization)]
    assert bns and all(l.momentum == 0.9 for l in bns)


def test_output_is_float32_under_mixed_precision():
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = build_unet("mobilenetv2", 3)
        assert model.get_layer("dec1_conv1").compute_dtype == "float16"
        assert model.output.dtype == "float32"
    finally:
        keras.mixed_precision.set_global_policy("float32")


def test_parameter_counts_match_the_saved_table():
    saved = json.loads((ROOT / "results" / "model_params.json").read_text())
    for encoder in ENCODERS:
        counted = count_parameters(build_unet(encoder, 3))
        assert counted == {key: saved[encoder][key] for key in counted}
