"""Run configurations.

    src/config_default.json   the frozen training defaults (git tag `defaults-frozen`)
    configs/Rxx.json          one small file per run: its id, seed and ONLY the settings that differ
                              from the defaults (so the factor under study is visible at a glance)
    results/Rxx/config.json   the fully resolved config a run actually used, plus provenance

`load_config("R04")` returns defaults + overrides. Unknown keys are rejected, so a typo cannot
silently fall back to a default.

Run from the project root to (re)write configs/R01.json ... R21.json from the table below:
    python -m src.config
"""
import json
from pathlib import Path

from src.data import ROOT

DEFAULT_JSON = Path(__file__).with_name("config_default.json")
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"

SEEDS = (0, 1, 2)

# (config name, study, overrides). Three seeds each -> R01 ... R21, in this order.
# Names match the "Config" column of the tracker's Runs tab.
CONFIG_TABLE = (
    ("Default", "E1, E2, E3 (shared)", {}),
    ("VH only", "E1", {"channels": "vh"}),
    ("VV+VH", "E1", {"channels": "vv_vh"}),
    ("BCE loss", "E2", {"loss": "bce"}),
    ("Dice loss", "E2", {"loss": "dice"}),
    ("Scratch U-Net", "E3", {"encoder": "scratch"}),
    ("MobileNetV2 + ImageNet", "E3", {"encoder": "mobilenetv2_imagenet"}),
)
CONFIG_NAMES = tuple(name for name, _, _ in CONFIG_TABLE)

# Which configs each study compares (the default is the shared reference of all three).
STUDIES = {
    "E1": {"question": "RQ1: which input channels?", "factor": "channels",
           "configs": ("VH only", "VV+VH", "Default")},
    "E2": {"question": "RQ2: which loss?", "factor": "loss",
           "configs": ("BCE loss", "Dice loss", "Default")},
    "E3": {"question": "RQ3: which encoder?", "factor": "encoder",
           "configs": ("Scratch U-Net", "Default", "MobileNetV2 + ImageNet")},
}

META_KEYS = ("run_id", "config_name", "study")

# Everything that defines how a model is trained or scored. After the freeze (task T19, git tag
# `defaults-frozen`) these files must not change; src/runner.check_frozen() enforces it.
FROZEN_TAG = "defaults-frozen"
FROZEN_PATHS = ("src/config_default.json", "configs", "src/config.py", "src/train.py", "src/models.py",
                "src/losses.py", "src/pipeline.py", "src/preprocess.py", "src/data.py", "src/evaluate.py",
                "src/metrics.py", "results/norm_stats.json")

# Display names of the setting values (used in tables and figures).
CHANNEL_LABELS = {"vh": "VH", "vv_vh": "VV+VH", "vv_vh_ratio": "VV+VH+ratio"}
LOSS_LABELS = {"bce": "BCE", "dice": "Dice", "bce_dice": "BCE+Dice"}
ENCODER_LABELS = {"scratch": "Scratch U-Net", "mobilenetv2": "MobileNetV2",
                  "mobilenetv2_imagenet": "MobileNetV2 + ImageNet"}


def default_config():
    return json.loads(DEFAULT_JSON.read_text())


def run_table():
    """[{run_id, config_name, study, seed, **overrides}, ...] for R01 ... R21."""
    runs = []
    for name, study, overrides in CONFIG_TABLE:
        for seed in SEEDS:
            runs.append({"run_id": f"R{len(runs) + 1:02d}", "config_name": name, "study": study,
                         "seed": seed, **overrides})
    return runs


RUN_IDS = tuple(r["run_id"] for r in run_table())


def runs_of(config_name):
    return [r["run_id"] for r in run_table() if r["config_name"] == config_name]


def resolve(overrides):
    """defaults + overrides -> full config dict. Rejects keys the defaults do not know."""
    cfg = default_config()
    unknown = set(overrides) - set(cfg) - set(META_KEYS)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return {**{k: overrides.get(k) for k in META_KEYS if k in overrides}, **cfg,
            **{k: v for k, v in overrides.items() if k not in META_KEYS}}


def load_config(run):
    """`run` is a run id ("R04") or the path of a JSON file with overrides."""
    path = Path(run)
    if path.suffix != ".json":
        path = CONFIG_DIR / f"{run}.json"
    cfg = resolve(json.loads(path.read_text()))
    cfg.setdefault("run_id", path.stem)
    return cfg


def run_dir(run_id):
    return RESULTS_DIR / run_id


def write_run_configs():
    CONFIG_DIR.mkdir(exist_ok=True)
    for run in run_table():
        (CONFIG_DIR / f"{run['run_id']}.json").write_text(json.dumps(run, indent=2) + "\n")
    return len(run_table())


if __name__ == "__main__":
    print(f"wrote {write_run_configs()} run configs to {CONFIG_DIR}")
