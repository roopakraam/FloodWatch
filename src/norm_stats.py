"""Per-channel mean/std on the training chips -> results/norm_stats.json.

Channels: VV, VH (clipped to [-35, 5] dB) and VV - VH (dB difference, computed after clipping).
Run from the project root:  python -m src.norm_stats
"""
import json

import numpy as np

from src.data import ROOT, list_split, load_chip

SPLIT = "train"
CLIP_DB = (-35.0, 5.0)
CHANNELS = ["VV", "VH", "VV_minus_VH"]
OUT_JSON = ROOT / "results" / "norm_stats.json"


def compute_channels(image):
    """(2, H, W) [VV, VH] dB image (NaNs kept) -> (3, H, W) [VV, VH, VV - VH], clipped first."""
    vv, vh = np.clip(image, *CLIP_DB)
    return np.stack([vv, vh, vv - vh]).astype(np.float32)


def main():
    pairs = list_split(SPLIT)
    # Per-channel buffers of non-NaN values; ~0.8 GB total for 252 chips.
    values = [[] for _ in CHANNELS]
    for s1_path, label_path in pairs:
        image, _ = load_chip(s1_path, label_path, nan_value=None)
        for c, ch in enumerate(compute_channels(image)):
            values[c].append(ch[~np.isnan(ch)])  # np.nanmean on the stacked values == ignoring NaN

    flat = [np.concatenate(v) for v in values]
    mean = [float(np.nanmean(x)) for x in flat]
    std = [float(np.nanstd(x)) for x in flat]

    out = {
        "split": SPLIT,
        "clip_db": list(CLIP_DB),
        "n_chips": len(pairs),
        "channels": CHANNELS,
        "mean": mean,
        "std": std,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
