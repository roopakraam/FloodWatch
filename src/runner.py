"""Start training runs as separate Python processes and show their logs. Used by the notebooks.

Why a separate process per run:
  * the laptop has a 4 GB GPU and WSL2 has about 7.5 GB of RAM; a training process needs roughly 4 GB of
    RAM, and TensorFlow never hands GPU or host memory back until the process ends. A fresh process per
    run keeps every run inside the same, known budget instead of growing over a long notebook session;
  * every run starts from the same clean state, so runs cannot influence each other;
  * if one run crashes, the notebook reports it and the finished runs are untouched.

The notebook stays the driver and the record: `run_training("R04")` streams the per-epoch lines into the
cell, and a run that is already finished is not repeated - its saved log is shown instead.

    logs/<run_id>.log          everything the process printed (including TensorFlow's own messages)
    results/<run_id>/          the run's outputs (see src/train.py)
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from src.config import FROZEN_PATHS, FROZEN_TAG, run_dir
from src.data import ROOT
from src.runlog import is_trained, replay, score_line

LOG_DIR = ROOT / "logs"


def _watch_memory(pid, peak, stop):
    """Sample the resident memory of the training process (and its children) twice a second."""
    try:
        proc = psutil.Process(pid)
        while not stop.is_set():
            rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
            peak[0] = max(peak[0], rss)
            stop.wait(0.5)
    except psutil.Error:
        pass


def run_training(config, evaluate=True, force=False, overrides=None, out_dir=None, log=print):
    """Train one run in a fresh process, streaming its log lines. Returns the run's metrics dict.

    config     run id ("R04") or path of a run config
    evaluate   also score test and Bolivia when training is done (once; see src/evaluate.py)
    overrides  dict of config values for an ad-hoc variant, e.g. {"epochs": 5}
    out_dir    output folder for an ad-hoc variant (default: results/<run_id>)
    """
    run_id = Path(str(config)).stem
    out = Path(out_dir) if out_dir is not None else run_dir(run_id)
    if is_trained(out) and not force:
        metrics = json.loads((out / "metrics.json").read_text())
        if not evaluate or all(split in metrics for split in ("test", "bolivia")):
            return replay(out, log, scores=("test", "bolivia") if evaluate else ())
    cmd = [sys.executable, "-m", "src.train", "--config", str(config)]
    if overrides:
        cmd += ["--set", *[f"{k}={json.dumps(v)}" for k, v in overrides.items()]]
    if out_dir is not None:
        cmd += ["--out", str(out.relative_to(ROOT))]
    if evaluate:
        cmd.append("--evaluate")
    if force:
        cmd.append("--force")

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{out.name if out_dir is not None else run_id}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    t0 = time.perf_counter()
    with open(log_path, "a") as log_file:
        log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd[1:])} =====\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=log_file, text=True, bufsize=1)
        peak, stop = [0], threading.Event()
        watcher = threading.Thread(target=_watch_memory, args=(proc.pid, peak, stop), daemon=True)
        watcher.start()
        for line in proc.stdout:
            log(line.rstrip("\n"))
            log_file.write(line)
        code = proc.wait()
        stop.set()
        watcher.join()

    if code != 0:
        tail = "".join(open(log_path).readlines()[-25:])
        raise RuntimeError(f"{run_id} failed (exit code {code}). Last lines of {log_path.relative_to(ROOT)}:\n{tail}")

    metrics = json.loads((out / "metrics.json").read_text())
    if peak[0] and "peak_host_ram_mb" not in metrics["training"] and time.perf_counter() - t0 > 30:
        # only a process that actually trained is worth recording (a replayed run exits within seconds)
        metrics["training"]["peak_host_ram_mb"] = round(peak[0] / 1024**2)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def run_evaluation(run_id, force=False, log=print):
    """Score a finished run on test and Bolivia in a fresh process (once; see src/evaluate.py)."""
    metrics = json.loads((run_dir(run_id) / "metrics.json").read_text())
    if not force and all(split in metrics for split in ("test", "bolivia")):
        log(f"[{run_id} was scored on test and Bolivia on {metrics['evaluated_utc'][:16].replace('T', ' ')} UTC; "
            f"not scoring it again]")
        for split in ("test", "bolivia"):
            log(score_line(run_id, split, metrics[split]))
        return metrics
    cmd = [sys.executable, "-m", "src.evaluate", "--run", run_id] + (["--force"] if force else [])
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{run_id}.log"
    with open(log_path, "a") as log_file:
        log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd[1:])} =====\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"}, stdout=subprocess.PIPE,
                                stderr=log_file, text=True, bufsize=1)
        for line in proc.stdout:
            log(line.rstrip("\n"))
            log_file.write(line)
        code = proc.wait()
    if code != 0:
        tail = "".join(open(log_path).readlines()[-25:])
        raise RuntimeError(f"evaluation of {run_id} failed (exit code {code}). Last lines of the log:\n{tail}")
    return json.loads((run_dir(run_id) / "metrics.json").read_text())


def run_benchmark(encoder="mobilenetv2", jit_compile=True, steps=30, channels="vv_vh_ratio", loss="bce_dice"):
    """Time one model alone on the GPU in a fresh process (see src/benchmark.py). Returns its result dict."""
    cmd = [sys.executable, "-m", "src.benchmark", "--encoder", encoder, "--channels", channels, "--loss", loss,
           "--steps", str(steps)]
    if not jit_compile:
        cmd.append("--no-jit")
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "benchmark.log", "a") as log_file:
        log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd[1:])} =====\n")
        log_file.flush()
        done = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=log_file, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"benchmark failed (exit code {done.returncode}); see logs/benchmark.log")
    return json.loads(done.stdout.strip().splitlines()[-1])


def check_frozen(tag=FROZEN_TAG):
    """Confirm that the training and scoring code is byte-identical to the frozen git tag.

    Returns the tagged commit id. Raises if the tag is missing or any frozen file differs from it, so an
    ablation notebook cannot silently run with settings that were changed after the freeze.
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    commit = git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
    if commit.returncode != 0:
        raise RuntimeError(f"git tag {tag!r} does not exist: freeze the defaults (task T19) before running ablations")
    changed = git("diff", "--name-only", tag, "--", *FROZEN_PATHS).stdout.split()
    if changed:
        raise RuntimeError(f"these files differ from the frozen tag {tag!r}: {changed}")
    return commit.stdout.strip()


def run_many(run_ids, evaluate=True, log=print):
    """Train several runs one after the other. Returns {run_id: metrics}."""
    results = {}
    for run_id in run_ids:
        log(f"\n{'=' * 100}")
        results[run_id] = run_training(run_id, evaluate=evaluate, log=log)
    return results
