# FloodWatch

Flood-water segmentation from Sentinel-1 radar with lightweight U-Nets, trained on the 446 hand-labelled
chips of [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) on a laptop with a 4 GB GPU.

The project asks how well a small network can map water in a single radar image, and which design
choices matter: the input channels (E1), the loss (E2) and the encoder (E3). Every configuration is
trained with three seeds, the test data are scored once per run, and every number in the report is
generated from a file under `results/`.

**Report:** [report/FloodWatch_report.pdf](report/FloodWatch_report.pdf)

<!-- results:start -->
## Results at a glance

Pooled water IoU, mean ± standard deviation over three seeds (generated from `results/` by `python -m src.report_assets`):

| Method | Validation IoU | Test IoU | Bolivia IoU (unseen flood) |
|:---|---:|---:|---:|
| Otsu threshold on VH (R00) | 0.388 | 0.433 | 0.653 |
| VH threshold tuned on the training labels (R00b) | 0.492 | 0.524 | 0.531 |
| **Default U-Net** (MobileNetV2, VV+VH+ratio, BCE+Dice; R01–R03) | 0.639 ± 0.009 | 0.667 ± 0.002 | 0.637 ± 0.017 |
| U-Net, VH only (R04–R06) | 0.612 ± 0.008 | 0.634 ± 0.007 | 0.623 ± 0.018 |
| U-Net, VV+VH (R07–R09) | 0.650 ± 0.009 | 0.670 ± 0.006 | 0.632 ± 0.038 |
| U-Net, BCE loss (R10–R12) | 0.616 ± 0.009 | 0.643 ± 0.011 | 0.559 ± 0.120 |
| U-Net, Dice loss (R13–R15) | 0.644 ± 0.007 | 0.660 ± 0.010 | 0.624 ± 0.019 |
| Scratch U-Net (R16–R18) | 0.641 ± 0.009 | 0.655 ± 0.010 | 0.671 ± 0.071 |
| MobileNetV2 + ImageNet weights (R19–R21) | 0.664 ± 0.007 | 0.662 ± 0.004 | 0.519 ± 0.019 |
<!-- results:end -->

What the numbers say, in short (the report gives the evidence and the caveats):

* On the test chips every U-Net is far ahead of both threshold rules.
* Adding VV to VH helps, and so does a Dice-based loss instead of BCE alone. The ratio channel, the size
  of the encoder and ImageNet weights make no measurable difference there.
* On Bolivia, a flood the models never saw, no network beats the Otsu threshold reliably, and the models
  with ImageNet weights are clearly worse than the same network trained from random weights.
* About a third of the best model's wrong pixels are labelled water that is not dark in the radar image.

## Repository layout

```
configs/        R01.json ... R21.json   one small file per run: seed + the one setting that differs from the default
data/           the dataset (not in git, see "Data")
figures/        figures written by the notebooks; figures/report/ holds the report figures
notebooks/      the project as it was carried out, in order (see "Notebooks")
report/         report source (Markdown), bibliography, published numbers, generated tables, PDF
results/        one folder per run + summary tables; the source of every reported number
src/            the library the notebooks and scripts use
tests/          unit tests (metrics, losses, pipeline, models, configs, error analysis, report tools)
```

`src/` in brief:

| File | What it does |
|---|---|
| `data.py` | paths, split lists, chip reading, cached splits |
| `preprocess.py` | the one preprocessing function (clip, channels, standardise, NaN → 0) shared by training, evaluation and the demo |
| `data_stats.py`, `norm_stats.py` | dataset checks and statistics; training-set mean and std |
| `metrics.py` | pooled water IoU / precision / recall / F1, mean per-chip IoU, bootstrap interval |
| `otsu_baseline.py` | the two threshold baselines (R00 Otsu, R00b tuned threshold) |
| `pipeline.py` | `tf.data` input pipeline: random crops, flips, rotations |
| `models.py` | U-Net with three encoders (scratch, MobileNetV2, MobileNetV2 + ImageNet) |
| `losses.py` | BCE, Dice, BCE+Dice and the water-IoU metric, all ignoring unlabelled pixels |
| `config.py`, `config_default.json` | frozen training defaults and the table of the 21 runs |
| `train.py`, `evaluate.py` | train one run; score a run on test and Bolivia (once) |
| `runner.py`, `runlog.py` | start runs as separate processes from a notebook; replay the log of finished runs |
| `benchmark.py` | controlled timing of a model (training step, inference, GPU memory) |
| `analysis.py`, `error_analysis.py` | tables over all runs, seed statistics, bootstrap; rule-based error categories |
| `make_figures.py`, `viz.py` | report figures; shared colours and plotting helpers |
| `report_assets.py`, `build_report.py` | numbers and tables for the report; Markdown → HTML → PDF |
| `check_report.py` | checks that every number in the report can be traced to a file or a source |
| `predict.py` | inference demo |

## Setup

Developed on Windows 11 with WSL2 (Ubuntu), Python 3.12, TensorFlow 2.21 and an NVIDIA RTX 3050 (4 GB).

```bash
python3 -m venv ~/tf-gpu-env
source ~/tf-gpu-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt           # exact versions used for the report
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # must list one GPU
python -m pytest                          # 67 tests, about a minute
```

All commands below are run from the project root with the environment active.

## Data

Only the hand-labelled Sentinel-1 chips, their labels and the split files are needed (about 700 MB).
The bucket is public; `gsutil` is part of the Google Cloud CLI.

```bash
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand    data/s1
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand data/labels
gsutil -m rsync -r gs://sen1floods11/v1.1/splits/flood_handlabeled                data/splits
gsutil cp gs://sen1floods11/v1.1/Sen1Floods11_Metadata.geojson                    data/metadata.geojson
```

The code expects short names for the four split files:

```bash
cd data/splits
mv flood_train_data.csv train.csv && mv flood_valid_data.csv val.csv
mv flood_test_data.csv test.csv   && mv flood_bolivia_data.csv bolivia.csv
cd ../..
python -m src.data_stats        # stops with an error unless there are 446 chips split 252 / 89 / 90 / 15
```

The first run that loads a split writes an uncompressed copy to `data/cache/`, which makes later loads
several times faster.

## Notebooks

The notebooks are the record of the project, in the order it was done. Each one is saved with its outputs.

| Notebook | Tasks | Content |
|---|---|---|
| `01_eda.ipynb` | T07–T10 | file and split checks, a look at the chips, training-set statistics, normalisation statistics |
| `02_otsu_baseline.ipynb` | T12 | Otsu threshold on VH (R00) and the tuned-threshold reference (R00b) |
| `03_pipeline_models_losses.ipynb` | T13–T15 | checks of the input pipeline, the three models, the losses; memorise-one-batch test |
| `04_smoke_test.ipynb` | T16 | 5-epoch smoke test: timing, memory, XLA, the BatchNorm-momentum problem |
| `05_default_run_R01.ipynb` | T17, T18 | the main run and its inspection on validation data |
| `06_E1_channels.ipynb` | T22 | input-channel study, runs R01–R09 |
| `07_E2_losses.ipynb` | T23 | loss study, runs R10–R15 |
| `08_E3_encoders.ipynb` | T24 | encoder study, runs R16–R21, and the cost of each encoder |
| `09_results_summary.ipynb` | T25 | all runs, mean ± std, which differences are real, bootstrap intervals, best model |
| `10_error_analysis.ipynb` | T26 | where and why the best model fails; test against Bolivia (RQ4) |
| `11_inference_demo.ipynb` | T31 | the command-line demo on chips not used in training |

A notebook that trains models starts each run in its own Python process and shows its log. A run that is
already finished is never repeated: its saved log is shown instead, so re-running a notebook is cheap and
cannot change a result. To re-run one from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/06_E1_channels.ipynb
```

## Train and evaluate without the notebooks

```bash
python -m src.otsu_baseline                              # R00 and R00b
python -m src.train --config configs/R04.json --evaluate # train R04, then score it once on test and Bolivia
python -m src.evaluate --run R04                         # score an already trained run (refuses to do it twice)
python -m src.analysis                                   # results/all_runs.csv and results/summary.csv
python -m src.make_figures                               # figures/report/*.png (--only <name> for one figure)
```

To repeat the whole project, run the notebooks in order (`01` to `11`); `04` to `08` train the 21 models,
about two and a half hours on the laptop named above. Then rebuild the figures and the report.

Each run writes to `results/<run>/`:

| File | Content |
|---|---|
| `config.json` | every setting the run used, plus the git commit, library versions and GPU |
| `history.csv`, `train.log` | per-epoch losses, IoU, learning rate and seconds |
| `metrics.json` | training summary and the validation, test and Bolivia scores |
| `per_chip.csv` | one row per chip and split: pixel counts, TP / FP / FN, per-chip scores |
| `best.keras` | the model at its best validation IoU (in git only for R19, the model the demo uses) |

`results/all_runs.csv` has one row per run in the layout of the project tracker's Runs tab.

## Inference demo

```bash
python src/predict.py --chip data/s1/Bolivia_103757_S1Hand.tif
python src/predict.py --chip <any 2-band VV/VH GeoTIFF in dB> --run R01 --out my_chip.png --tif my_mask.tif
```

The demo loads the best-validation model (`results/best_run.json`, R19; its weights are in the repository), applies the same preprocessing as
training, and saves a picture of the radar image next to the predicted water mask. If a hand label for
the chip exists it also shows the label, an error map and the IoU. `--tif` writes the mask as a
georeferenced GeoTIFF.

## Rebuild the report

```bash
python -m src.make_figures      # figures/report/*.png from results/
python -m src.report_assets     # numbers and tables from results/  ->  report/generated/
python -m src.build_report      # report/report.md + report/sections/*.md  ->  HTML  ->  PDF
python -m src.check_report      # every number in the report traced? (--list shows the typed ones)
```

The report text contains placeholders instead of typed results; the build fails if one cannot be filled.
`report/generated/sources.json` names the result file behind every generated number. The few numbers
that are typed by hand (values from the literature, facts of the design) are registered in
`report/typed_numbers.csv` with the source each was checked against, and `report/published_results.csv`
records where every published score was read. The PDF is printed by Chrome or Edge in headless mode
(set `CHROME_PATH` if neither is found).

## Rules the experiments followed

1. Every choice (stopping, checkpoint, "best" model) is made on the validation chips.
2. Test and Bolivia are scored once per finished run; `src/evaluate.py` refuses to score a run twice.
3. Within a study only the factor under study changes. The training defaults were frozen and tagged
   (`git tag defaults-frozen`) before any ablation run; each ablation notebook first checks that the
   training and scoring code is identical to that tag.

## Notes on the hardware

* A training process needs about 4.4 GB of RAM and 0.6–1 GB of GPU memory, so one run at a time fits a
  machine with 8 GB of RAM in WSL and a 4 GB GPU.
* XLA compilation makes a training step about seven times faster on this machine. The compiled kernels
  are cached in `~/.cache/floodwatch/xla` (override with `FLOODWATCH_XLA_CACHE`); the first run of each
  architecture spends a few minutes compiling.
* Keras' MobileNetV2 uses a BatchNorm momentum of 0.999. With 126 updates per epoch that makes the
  validation score useless for many epochs; all models here use 0.99 (see `notebooks/04_smoke_test.ipynb`).

## Data and citation

Sen1Floods11: D. Bonafilia, B. Tellman, T. Anderson, E. Issenberg. *Sen1Floods11: A Georeferenced Dataset
to Train and Test Deep Learning Flood Algorithms for Sentinel-1.* CVPR Workshops, 2020.
