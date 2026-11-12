## Appendix A. All runs {.unnumbered}

{{table:allruns}}

## Appendix B. Reproducing the results {.unnumbered}

The code, the run configurations, the result files and the notebooks are in the project repository; `README.md` gives the set-up and the commands. In brief:

- **Data.** The hand-labelled Sentinel-1 chips, their labels and the split files of Sen1Floods11 v1.1, downloaded from the public bucket `gs://sen1floods11`.
- **Runs.** `configs/R01.json` to `configs/R21.json` hold the seed and the one setting in which each run differs from `src/config_default.json`. `python -m src.train --config configs/R04.json --evaluate` trains a run and scores it once on test and Bolivia; the notebooks `04` to `08` do the same for all runs and keep their logs.
- **Frozen defaults.** The training defaults were tagged `defaults-frozen` in git before any ablation run. All ablation runs were produced by that version of the training and scoring code; each run's `config.json` records the commit.
- **Numbers.** Every number in this report is read at build time from a file under `results/` by `python -m src.report_assets`; `report/generated/sources.json` lists the file and field behind each one. `results/all_runs.csv` is the table of all runs.
- **Software and hardware.** Python 3.12, TensorFlow {{tf_version}}, Keras {{keras_version}}; {{gpu_name}} with 4 GB under WSL2. Exact package versions are in `requirements.txt`.
