"""Tests for the TensorFlow-free tooling: report builder, number check, seed statistics, run log, demo padding."""
import numpy as np
import pandas as pd
import pytest

from src import analysis, check_report
from src.build_report import Builder, format_authors, parse_bibtex
from src.predict import predict_proba
from src.runlog import format_epoch

BIB = """
@inproceedings{sandler2018, author = {Sandler, Mark and Chen, Liang-Chieh}, title = {{MobileNetV2}: Inverted {R}esiduals},
  booktitle = {CVPR}, year = {2018}, pages = {4510--4520}}
@misc{many2023, author = {Jakubik, Johannes and others}, title = {Foundation Models}, year = {2023}}
"""


def test_bibtex_parser_and_author_format():
    bib = parse_bibtex(BIB)
    assert set(bib) == {"sandler2018", "many2023"}
    assert bib["sandler2018"]["title"] == "{MobileNetV2}: Inverted {R}esiduals" and bib["sandler2018"]["pages"] == "4510--4520"
    assert format_authors(bib["sandler2018"]["author"]) == "M. Sandler and L.-C. Chen"
    assert format_authors(bib["many2023"]["author"]) == "J. Jakubik et al"


def test_builder_numbers_sections_figures_and_citations():
    text = ("## Intro {#sec:intro}\n\nSee @sec:method and @tbl:t [@bonafilia2020sen1floods11; @otsu1979threshold].\n\n"
            "## Method {#sec:method}\n\nTable: A caption {#tbl:t}\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
            "As in @sec:intro [@otsu1979threshold].\n\n## References {.unnumbered}\n\n{{references}}\n")
    b = Builder(text)
    _, body = b.to_html()
    assert b.labels == {"sec:intro": "Section 1", "sec:method": "Section 2", "tbl:t": "Table 1"}
    assert b.cited == ["bonafilia2020sen1floods11", "otsu1979threshold"]          # numbered in order of first use
    assert ">Section 2</a>" in body and ">Table 1</a>" in body and body.count('href="#ref-otsu1979threshold"') == 2
    assert "<h2>References</h2>" in body and '<span class="secnum">2</span>' in body


@pytest.mark.parametrize("text", ["{{no_such_number}}", "See @fig:missing.", "Cite [@nobody2099]."])
def test_builder_stops_on_anything_it_cannot_resolve(text):
    with pytest.raises(SystemExit):
        Builder(text + "\n\n{{references}}\n").to_html()


def test_typed_numbers_are_found_and_names_with_digits_are_not():
    text = ("Sentinel-1 chips of 512×512 pixels [@yadav2022attentive] reach 0.672 (run R01, MobileNetV2, F1, RQ3); "
            "{{default_test_iou_pm}} against −23.60 dB, 4,831 chips, 95% and `python3.12` in @tbl:main; $x_1 = 5$.")
    assert check_report.typed_numbers(text) == ["512", "512", "0.672", "−23.60", "4,831", "95%"]
    assert check_report.typed_numbers("1. First item of a list with 3 seeds.") == ["3"]


def test_every_number_in_the_report_is_traced():
    """The report on disk: each placeholder has a source file, each typed number is in report/typed_numbers.csv."""
    files = check_report.source_files()
    assert check_report.check_generated(files) == []
    assert check_report.check_typed(files) == []


def test_mean_std_text():
    assert analysis.mean_std(0.66666, 0.00234) == "0.667 ± 0.002"
    assert analysis.mean_std(0.4334, float("nan")) == "0.433"
    assert analysis.mean_std(float("nan"), float("nan")) == ""


def test_gap_table_applies_the_gap_versus_std_rule():
    rows = []
    for config, values in (("VH only", (0.60, 0.61, 0.62)), ("VV+VH", (0.64, 0.66, 0.65)), ("Default", (0.650, 0.655, 0.652))):
        rows += [{"Run ID": f"{config}{i}", "Config": config, "Status": "Finished", "Test IoU": v, "Bolivia IoU": v}
                 for i, v in enumerate(values)]
    gaps = analysis.gap_table("E1", pd.DataFrame(rows), metrics=("Test IoU",)).set_index(["A", "B"])
    assert bool(gaps.loc[("VH only", "Default"), "Gap larger than std"])            # 0.04 apart, std 0.01
    assert not bool(gaps.loc[("VV+VH", "Default"), "Gap larger than std"])          # 0.002 apart, std 0.01
    assert np.isclose(gaps.loc[("VH only", "VV+VH"), "Gap (A - B)"], -0.04)


def test_epoch_line():
    row = {"epoch": 2, "loss": 0.5, "water_iou": 0.61, "val_loss": 0.45, "val_water_iou": 0.63, "learning_rate": 1e-3,
           "seconds": 8.44, "is_best": 1}
    assert format_epoch(row, 60) == "epoch  3/60 | loss 0.5000  iou 0.6100 | val_loss 0.4500  val_iou 0.6300 | lr 1.0e-03 |   8.4s *"


class _FakeModel:
    """Stands in for a Keras model: returns the mean of the input channels as 'probability'."""
    def predict_on_batch(self, x):
        self.seen = x.shape
        return x.mean(axis=-1, keepdims=True)


def test_demo_pads_to_a_multiple_of_32_and_crops_back():
    image = np.random.default_rng(0).uniform(-30, 0, size=(2, 70, 100)).astype(np.float32)
    model = _FakeModel()
    prob = predict_proba(model, {"channels": "vv_vh_ratio"}, image)
    assert model.seen == (1, 96, 128, 3)       # padded up to the next multiples of 32
    assert prob.shape == (70, 100) and prob.dtype == np.float32
