"""Input pipeline: splits held in RAM -> tf.data datasets.

All 446 chips fit in memory (about 1 GB raw), so a split is loaded once, pushed through
src.preprocess.preprocess, and tf.data is built on top of the NumPy arrays.

Training   : random 256x256 crop + random horizontal / vertical flip + random 90-degree rotation,
             the SAME transform applied to image and mask; `crops_per_chip` (4) crops of every chip
             per epoch, shuffled, batch 8  ->  252 * 4 / 8 = 126 steps per epoch.
Evaluation : full 512x512 chips, fixed order, no augmentation.

Labels keep -1 (no data) all the way to the loss and the metrics, which mask those pixels.
The augmentation stream is driven by one NumPy generator seeded with the run's seed, so the sequence
of crops is reproducible.
"""
from functools import lru_cache

import numpy as np
import tensorflow as tf

from src.data import load_split
from src.preprocess import CHANNEL_SETS, load_norm_stats, preprocess

CROP = 256
CROPS_PER_CHIP = 4
BATCH = 8


@lru_cache(maxsize=None)
def _split_arrays(split):
    """Standardised 3-channel inputs and labels of one split, loaded once per process."""
    data = load_split(split)
    norm = load_norm_stats()
    n, _, h, w = data.images.shape
    x = np.empty((n, h, w, 3), dtype=np.float32)
    for i in range(n):  # chip by chip: no multi-GB temporaries
        x[i] = preprocess(data.images[i], "vv_vh_ratio", norm)
    x.setflags(write=False)
    y = data.labels[..., None]  # (N, H, W, 1) int8, a view
    y.setflags(write=False)
    return x, y


def load_arrays(split, channels="vv_vh_ratio"):
    """X (N, 512, 512, C) float32 standardised with NaN -> 0, Y (N, 512, 512, 1) int8 in {-1, 0, 1}.

    The channel subsets are views of one shared array, so asking for another channel set is free.
    """
    x, y = _split_arrays(split)
    return x[..., CHANNEL_SETS[channels]], y


def augment(x, y, rng, crop=CROP):
    """Random crop, flips and 90-degree rotation, identical for image `x` (H, W, C) and mask `y` (H, W, 1)."""
    h, w = y.shape[:2]
    top = int(rng.integers(0, h - crop + 1))
    left = int(rng.integers(0, w - crop + 1))
    x = x[top:top + crop, left:left + crop]
    y = y[top:top + crop, left:left + crop]
    if rng.random() < 0.5:  # horizontal flip
        x, y = x[:, ::-1], y[:, ::-1]
    if rng.random() < 0.5:  # vertical flip
        x, y = x[::-1], y[::-1]
    k = int(rng.integers(0, 4))  # 0, 90, 180 or 270 degrees
    x, y = np.rot90(x, k), np.rot90(y, k)
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y, dtype=np.float32)


def make_train_dataset(X, Y, batch=BATCH, crops_per_chip=CROPS_PER_CHIP, crop=CROP, seed=0):
    """One pass = `crops_per_chip` random augmented crops of every chip, in random order."""
    rng = np.random.default_rng(seed)
    c = X.shape[-1]

    def gen():  # called again at the start of every epoch; `rng` keeps advancing, so epochs differ
        order = np.repeat(np.arange(len(X)), crops_per_chip)
        rng.shuffle(order)
        for i in order:
            yield augment(X[i], Y[i], rng, crop)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((crop, crop, c), tf.float32),
            tf.TensorSpec((crop, crop, 1), tf.float32),
        ),
    )
    return ds.batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def make_eval_dataset(X, Y, batch=BATCH):
    """Full chips, no augmentation, fixed order."""
    h, w, c = X.shape[1:]

    def gen():
        for i in range(len(X)):
            yield X[i], Y[i].astype(np.float32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((h, w, c), tf.float32),
            tf.TensorSpec((h, w, 1), tf.float32),
        ),
    )
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def steps_per_epoch(n_chips, batch=BATCH, crops_per_chip=CROPS_PER_CHIP):
    return (n_chips * crops_per_chip) // batch
