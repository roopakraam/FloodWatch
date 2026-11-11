"""Per-channel mean / std on the TRAINING chips only -> results/norm_stats.json.

Channels: VV and VH clipped to [-35, 5] dB, and VV - VH computed from the clipped values (the same
`compute_channels` every model input goes through). NaN pixels are ignored. Sums are accumulated in
float64, so the result does not depend on chip order or on float32 round-off.

Only the training split is touched: statistics that feed the model must not see validation or test data.

Run from the project root:  python -m src.norm_stats
"""
import json

import numpy as np

from src.data import load_split
from src.preprocess import CHANNELS, CLIP_DB, NORM_JSON, compute_channels

SPLIT = "train"


def compute_norm_stats(images):
    """(N, 2, H, W) raw dB images -> dict with per-channel mean / std / pixel count (NaN ignored)."""
    n = np.zeros(len(CHANNELS), dtype=np.int64)
    s1 = np.zeros(len(CHANNELS), dtype=np.float64)
    s2 = np.zeros(len(CHANNELS), dtype=np.float64)
    for image in images:  # chip by chip keeps memory flat
        ch = compute_channels(image).astype(np.float64).reshape(len(CHANNELS), -1)
        ok = ~np.isnan(ch)
        ch = np.where(ok, ch, 0.0)
        n += ok.sum(axis=1)
        s1 += ch.sum(axis=1)
        s2 += (ch * ch).sum(axis=1)
    mean = s1 / n
    std = np.sqrt(s2 / n - mean**2)  # population std, as np.nanstd
    return {"mean": mean.tolist(), "std": std.tolist(), "n_pixels": n.tolist()}


def main():
    data = load_split(SPLIT)
    out = {
        "split": SPLIT,
        "clip_db": list(CLIP_DB),
        "n_chips": len(data.names),
        "channels": list(CHANNELS),
        **compute_norm_stats(data.images),
    }
    NORM_JSON.parent.mkdir(exist_ok=True)
    NORM_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("wrote", NORM_JSON)


if __name__ == "__main__":
    main()
