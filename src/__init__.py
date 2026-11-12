"""FloodWatch: flood-water segmentation from Sentinel-1 SAR (Sen1Floods11) with lightweight U-Nets.

Importing this package (before TensorFlow is imported) switches on a persistent XLA compilation cache.
Why: on this machine (RTX 3050 under WSL2) a training step takes ~65 ms when the model is compiled
with XLA and ~860 ms when it is not, but compiling a model takes about 4 minutes. With the cache the
compiled kernels are stored on disk, so only the first run of each architecture pays that cost.
The cache holds machine code only; it cannot change any result. Delete the folder to reset it.

    FLOODWATCH_XLA_CACHE   cache folder (default: ~/.cache/floodwatch/xla)
"""
import os as _os

if "TF_XLA_FLAGS" not in _os.environ:
    _cache = _os.environ.get("FLOODWATCH_XLA_CACHE", _os.path.expanduser("~/.cache/floodwatch/xla"))
    _os.makedirs(_cache, exist_ok=True)
    _os.environ["TF_XLA_FLAGS"] = f"--tf_xla_persistent_cache_directory={_cache}"
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's INFO / WARNING start-up chatter
