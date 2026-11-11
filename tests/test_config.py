"""Tests for src/config.py: the 21 planned runs and the one-factor-at-a-time rule."""
import json

import pytest

from src.config import CONFIG_DIR, CONFIG_NAMES, RUN_IDS, STUDIES, default_config, load_config, resolve, run_table


def test_twenty_one_runs_seven_configs_three_seeds():
    runs = run_table()
    assert [r["run_id"] for r in runs] == [f"R{i:02d}" for i in range(1, 22)] == list(RUN_IDS)
    for name in CONFIG_NAMES:
        assert sorted(r["seed"] for r in runs if r["config_name"] == name) == [0, 1, 2]


def test_every_run_changes_at_most_one_factor():
    default = default_config()
    for run_id in RUN_IDS:
        cfg = load_config(run_id)
        changed = {k for k in default if k != "seed" and cfg[k] != default[k]}
        assert len(changed) <= 1, (run_id, changed)
        if cfg["config_name"] == "Default":
            assert not changed


def test_studies_only_vary_their_own_factor():
    default = default_config()
    for study in STUDIES.values():
        for name in study["configs"]:
            cfg = load_config(next(r["run_id"] for r in run_table() if r["config_name"] == name))
            changed = {k for k in default if k != "seed" and cfg[k] != default[k]}
            assert changed <= {study["factor"]}


def test_config_files_on_disk_match_the_table():
    for run in run_table():
        assert json.loads((CONFIG_DIR / f"{run['run_id']}.json").read_text()) == run


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError):
        resolve({"learning_rat": 0.01})
