"""Data loading helpers for the Sen1Floods11 hand-labelled Sentinel-1 chips."""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
S1_DIR = DATA_DIR / "s1"
LABEL_DIR = DATA_DIR / "labels"
SPLIT_DIR = DATA_DIR / "splits"

SPLITS = ("train", "val", "test", "bolivia")

# Label values in the *LabelHand.tif files.
NO_DATA = -1
NON_WATER = 0
WATER = 1


def list_split(split):
    """Return [(s1_path, label_path), ...] for a split CSV.

    `split` is one of SPLITS. The CSVs have no header row; each row is
    `<chip>_S1Hand.tif,<chip>_LabelHand.tif`.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv", header=None, names=["s1", "label"])
    return [(S1_DIR / s1, LABEL_DIR / label) for s1, label in zip(df["s1"], df["label"])]


def load_chip(s1_path, label_path, nan_value=0.0):
    """Read one chip and its label with rasterio.

    Returns
    -------
    image : float32 array, shape (2, H, W), bands [VV, VH] (dB).
        NaNs (missing SAR pixels) are replaced with `nan_value`; pass
        `nan_value=None` to keep them.
    label : int16 array, shape (H, W), values in {NO_DATA (-1), NON_WATER (0), WATER (1)}.
    """
    with rasterio.open(s1_path) as src:
        if src.count != 2:
            raise ValueError(f"{s1_path} has {src.count} bands, expected 2 (VV, VH)")
        image = src.read().astype(np.float32)
    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.int16)

    if image.shape[1:] != label.shape:
        raise ValueError(f"shape mismatch: {s1_path} {image.shape[1:]} vs {label_path} {label.shape}")
    if nan_value is not None:
        image = np.nan_to_num(image, nan=nan_value)
    return image, label


def load_split(split, **kwargs):
    """Load every chip in a split. Returns (images, labels) as stacked arrays."""
    pairs = list_split(split)
    chips = [load_chip(s1, lab, **kwargs) for s1, lab in pairs]
    return np.stack([c[0] for c in chips]), np.stack([c[1] for c in chips])
