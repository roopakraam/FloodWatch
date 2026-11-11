"""Input pipeline: chips -> standardised channels -> tf.data datasets.

Channel sets (a setting, so the channel study can switch inputs without new code):
    "vh"          -> [VH]
    "vv_vh"       -> [VV, VH]
    "vv_vh_ratio" -> [VV, VH, VV - VH]      (default)

Per chip: read VV, VH, label -> clip to [-35, 5] dB -> build channels -> standardise with
results/norm_stats.json (train-set mean/std) -> replace NaN with 0 (= the mean after standardising).
The label keeps -1 (no data) so the loss can mask those pixels.

Training: random 256x256 crops + random flips + random 90-degree rotations, 4 crops per
chip per epoch, batch 8. Validation / test: full 512x512 chips, no augmentation.

Run from the project root to plot one training batch:  python -m src.pipeline
"""
import json

import numpy as np
import tensorflow as tf

from src.data import ROOT, list_split, load_chip
from src.norm_stats import CHANNELS, compute_channels

NORM_JSON = ROOT / "results" / "norm_stats.json"

# indices into norm_stats' [VV, VH, VV_minus_VH]
CHANNEL_SETS = {
    "vh": [1],
    "vv_vh": [0, 1],
    "vv_vh_ratio": [0, 1, 2],
}

CROP = 256
CROPS_PER_CHIP = 4
BATCH = 8


def load_norm_stats():
    stats = json.loads(NORM_JSON.read_text())
    assert stats["channels"] == CHANNELS, "norm_stats.json channel order changed"
    return np.array(stats["mean"], dtype=np.float32), np.array(stats["std"], dtype=np.float32)


def prepare_chip(s1_path, label_path, channels="vv_vh_ratio", norm=None):
    """One chip -> (image (H, W, C) float32 standardised, NaN -> 0; label (H, W, 1) float32 in {-1, 0, 1})."""
    idx = CHANNEL_SETS[channels]
    mean, std = norm if norm is not None else load_norm_stats()

    image, label = load_chip(s1_path, label_path, nan_value=None)  # keep NaNs for now
    x = compute_channels(image)[idx]  # clip first, then build channels
    x = (x - mean[idx, None, None]) / std[idx, None, None]
    x = np.nan_to_num(x, nan=0.0)  # after standardising, so 0 means "average pixel"
    return x.transpose(1, 2, 0).astype(np.float32), label[..., None].astype(np.float32)


def load_arrays(split, channels="vv_vh_ratio"):
    """Load a whole split into RAM: X (N, 512, 512, C), Y (N, 512, 512, 1). ~0.8 GB for train."""
    norm = load_norm_stats()
    pairs = list_split(split)
    xs, ys = zip(*(prepare_chip(s, l, channels, norm) for s, l in pairs))
    return np.stack(xs), np.stack(ys)


def augment(x, y, rng, crop=CROP):
    """Same random crop, flips and 90-degree rotation applied to image and label together."""
    h, w = y.shape[:2]
    top = rng.integers(0, h - crop + 1)
    left = rng.integers(0, w - crop + 1)
    x = x[top:top + crop, left:left + crop]
    y = y[top:top + crop, left:left + crop]
    if rng.random() < 0.5:
        x, y = x[:, ::-1], y[:, ::-1]
    if rng.random() < 0.5:
        x, y = x[::-1], y[::-1]
    k = int(rng.integers(0, 4))
    x, y = np.rot90(x, k), np.rot90(y, k)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def make_train_dataset(X, Y, batch=BATCH, crops_per_chip=CROPS_PER_CHIP, crop=CROP, seed=0):
    """Each epoch: every chip gives `crops_per_chip` random crops, shuffled, in batches."""
    rng = np.random.default_rng(seed)
    c = X.shape[-1]

    def gen():
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
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def make_eval_dataset(X, Y, batch=BATCH):
    """Full chips, no augmentation, fixed order."""
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def plot_batch(images, labels, channels, path):
    """Show the VH channel next to its mask; -1 (no data) gets its own colour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    vh = CHANNEL_SETS[channels].index(1)
    cmap = ListedColormap(["lightgrey", "saddlebrown", "royalblue"])  # -1, 0, 1
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.5 * n))
    for row, img, lab in zip(axes, images, labels):
        row[0].imshow(img[..., vh], cmap="gray", vmin=-3, vmax=3)
        row[0].set_title("VH (standardised)", fontsize=8)
        row[1].imshow(lab[..., 0], cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        row[1].set_title("label (grey=no data, blue=water)", fontsize=8)
        row[0].axis("off")
        row[1].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)


def main(channels="vv_vh_ratio"):
    X, Y = load_arrays("train", channels)
    print("train arrays", X.shape, Y.shape, "| NaN left:", int(np.isnan(X).sum()))
    print("label values:", np.unique(Y))
    print("per-channel mean/std after standardising:", X.mean(axis=(0, 1, 2)).round(3), X.std(axis=(0, 1, 2)).round(3))

    ds = make_train_dataset(X, Y)
    imgs, labs = next(iter(ds))
    imgs, labs = imgs.numpy(), labs.numpy()
    print("train batch", imgs.shape, labs.shape)
    print("steps per epoch:", sum(1 for _ in ds))

    vh = CHANNEL_SETS[channels].index(1)
    water = labs[..., 0] == 1
    dry = labs[..., 0] == 0
    print(f"alignment check - mean VH on water {imgs[..., vh][water].mean():.2f} vs non-water {imgs[..., vh][dry].mean():.2f} (water should be lower)")

    Xv, Yv = load_arrays("val", channels)
    ev_imgs, ev_labs = next(iter(make_eval_dataset(Xv, Yv)))
    print("val batch", ev_imgs.shape, ev_labs.shape)

    out = ROOT / "figures" / "pipeline_batch.png"
    plot_batch(imgs[:6], labs[:6], channels, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
