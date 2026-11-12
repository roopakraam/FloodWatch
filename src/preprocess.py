"""The one preprocessing function shared by training, evaluation and the inference demo.

    raw chip (2, H, W) [VV, VH] in dB, NaN = missing
      -> clip both bands to [-35, 5] dB
      -> build channels [VV, VH, VV - VH]   (the "ratio": a division in linear units is a subtraction in dB)
      -> keep the channels the config asks for
      -> standardise with the TRAINING-set mean / std from results/norm_stats.json
      -> NaN -> 0  (after standardising, 0 is the average training pixel)
      -> (H, W, C) float32

Channel sets (a config setting, so the channel study E1 needs no new code):
    "vh"          -> [VH]
    "vv_vh"       -> [VV, VH]
    "vv_vh_ratio" -> [VV, VH, VV - VH]   (default)
"""
import json

import numpy as np

from src.data import ROOT

CLIP_DB = (-35.0, 5.0)
CHANNELS = ("VV", "VH", "VV_minus_VH")
# Slices (not index lists) into CHANNELS, so selecting channels from an array is a view, not a copy.
CHANNEL_SETS = {
    "vh": slice(1, 2),
    "vv_vh": slice(0, 2),
    "vv_vh_ratio": slice(0, 3),
}
NORM_JSON = ROOT / "results" / "norm_stats.json"


def n_channels(channels):
    s = CHANNEL_SETS[channels]
    return s.stop - s.start


def channel_names(channels):
    return CHANNELS[CHANNEL_SETS[channels]]


def compute_channels(image):
    """(..., 2, H, W) [VV, VH] dB (NaN kept) -> (..., 3, H, W) [VV, VH, VV - VH]. Clips first."""
    image = np.clip(np.asarray(image, dtype=np.float32), *CLIP_DB)  # NaN stays NaN
    vv, vh = image[..., 0, :, :], image[..., 1, :, :]
    return np.stack([vv, vh, vv - vh], axis=-3)


def load_norm_stats(path=NORM_JSON):
    """(mean, std) float32 arrays for CHANNELS, computed on the training split by src/norm_stats.py."""
    stats = json.loads(path.read_text())
    if tuple(stats["channels"]) != CHANNELS or tuple(stats["clip_db"]) != CLIP_DB:
        raise ValueError(f"{path} was computed with different channels / clip range; re-run src.norm_stats")
    return np.array(stats["mean"], dtype=np.float32), np.array(stats["std"], dtype=np.float32)


def preprocess(image, channels="vv_vh_ratio", norm=None):
    """Raw dB chip(s) (..., 2, H, W) -> model input (..., H, W, C) float32, standardised, NaN -> 0."""
    sel = CHANNEL_SETS[channels]
    mean, std = norm if norm is not None else load_norm_stats()
    x = compute_channels(image)[..., sel, :, :]
    x = (x - mean[sel, None, None]) / std[sel, None, None]
    x[np.isnan(x)] = 0.0
    return np.ascontiguousarray(np.moveaxis(x, -3, -1), dtype=np.float32)


def finite_mask(image):
    """(..., 2, H, W) -> (..., H, W) bool: True where both SAR bands were actually observed."""
    return np.isfinite(np.asarray(image)).all(axis=-3)
