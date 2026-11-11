"""Paths, split lists and chip I/O for the hand-labelled Sen1Floods11 Sentinel-1 chips.

Layout expected under data/ (not tracked by git, see README for the download commands):

    data/s1/<Event>_<id>_S1Hand.tif         2 bands, float32 dB: band 1 = VV, band 2 = VH (NaN = no SAR data)
    data/labels/<Event>_<id>_LabelHand.tif  1 band, int16: -1 = no data, 0 = not water, 1 = water
    data/splits/{train,val,test,bolivia}.csv   no header, one "<s1 file>,<label file>" pair per row

Reading 446 deflate-compressed GeoTIFFs from the Windows drive takes ~45 s, so `load_split` keeps an
uncompressed copy of each split in data/cache/<split>.npz. The cache holds the RAW values (dB with NaN,
labels with -1); every bit of preprocessing lives in src/preprocess.py so there is one code path for
training, evaluation and the inference demo.
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
S1_DIR = DATA_DIR / "s1"
LABEL_DIR = DATA_DIR / "labels"
SPLIT_DIR = DATA_DIR / "splits"
CACHE_DIR = DATA_DIR / "cache"

SPLITS = ("train", "val", "test", "bolivia")
SPLIT_SIZES = {"train": 252, "val": 89, "test": 90, "bolivia": 15}  # official v1.1 hand-labelled splits
BANDS = ("VV", "VH")
CHIP_SIZE = 512

# Label values in the *LabelHand.tif files.
NO_DATA = -1
NON_WATER = 0
WATER = 1


class SplitData(NamedTuple):
    """One split held in RAM."""

    names: list            # chip ids, e.g. "Ghana_103272", in split-CSV order
    images: np.ndarray     # (N, 2, 512, 512) float32 dB, bands [VV, VH], NaN where the SAR image is missing
    labels: np.ndarray     # (N, 512, 512) int8 in {-1, 0, 1}


def chip_name(path):
    """'.../Ghana_103272_S1Hand.tif' -> 'Ghana_103272'."""
    name = Path(path).name
    for suffix in ("_S1Hand.tif", "_LabelHand.tif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def event_of(name):
    """'Ghana_103272' -> 'Ghana' (the flood event / country the chip was cut from)."""
    return chip_name(name).rsplit("_", 1)[0]


def list_split(split):
    """Return [(s1_path, label_path), ...] for a split CSV, in file order."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv", header=None, names=["s1", "label"])
    return [(S1_DIR / s1, LABEL_DIR / label) for s1, label in zip(df["s1"], df["label"])]


def read_image(s1_path):
    """One S1Hand chip -> (2, H, W) float32 dB, bands [VV, VH]. NaN (missing SAR pixels) is kept."""
    with rasterio.open(s1_path) as src:
        if src.count != 2:
            raise ValueError(f"{s1_path} has {src.count} bands, expected 2 (VV, VH)")
        return src.read().astype(np.float32)


def read_label(label_path):
    """One LabelHand chip -> (H, W) int8 in {-1, 0, 1}."""
    with rasterio.open(label_path) as src:
        label = src.read(1)
    values = np.unique(label)
    if not np.isin(values, (NO_DATA, NON_WATER, WATER)).all():
        raise ValueError(f"{label_path} holds unexpected label values {values}")
    return label.astype(np.int8)


def load_chip(s1_path, label_path, nan_value=None):
    """Read one chip and its label.

    Returns (image (2, H, W) float32, label (H, W) int8). NaNs are kept unless `nan_value` is given;
    model inputs should go through src.preprocess.preprocess instead of using `nan_value`.
    """
    image, label = read_image(s1_path), read_label(label_path)
    if image.shape[1:] != label.shape:
        raise ValueError(f"shape mismatch: {s1_path} {image.shape[1:]} vs {label_path} {label.shape}")
    if nan_value is not None:
        image = np.nan_to_num(image, nan=nan_value)
    return image, label


def _read_split(split):
    pairs = list_split(split)
    images = np.empty((len(pairs), 2, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    labels = np.empty((len(pairs), CHIP_SIZE, CHIP_SIZE), dtype=np.int8)
    for i, (s1_path, label_path) in enumerate(pairs):
        images[i], labels[i] = load_chip(s1_path, label_path)
    return SplitData([chip_name(s1) for s1, _ in pairs], images, labels)


def load_split(split, use_cache=True):
    """Load a whole split into RAM (raw values). Cached in data/cache/<split>.npz after the first call."""
    names = [chip_name(s1) for s1, _ in list_split(split)]
    cache = CACHE_DIR / f"{split}.npz"
    if use_cache and cache.exists():
        with np.load(cache) as z:
            if z["names"].tolist() == names:  # the split CSV still lists the same chips, in the same order
                return SplitData(names, z["images"], z["labels"])
    data = _read_split(split)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, names=np.array(names), images=data.images, labels=data.labels)
    return data
