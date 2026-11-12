"""Check that every number in the report can be traced (task T32).

    python -m src.check_report            # exit code 1 if anything is untraced
    python -m src.check_report --list     # also print every typed number with the lines it appears on

A number reaches the report in one of two ways, and each has its own check.

1. Generated numbers. The text holds a placeholder such as {{default_test_iou_pm}}; src/report_assets.py
   fills it from a file under results/ and records that file in report/generated/sources.json. The check:
   every placeholder has a value and a source, and the file named in the source exists.

2. Typed numbers. Values from the literature (0.672 for the Attentive U-Net) and facts of the design
   (512x512 pixels) are written in the text by hand. They are registered in report/typed_numbers.csv
   with what they mean and where they were checked. The check: every number typed in the text is in
   the register, and the register holds nothing that the text no longer uses.

Names that merely contain digits (Sentinel-1, MobileNetV2, R01, RQ3, F1) are not numbers and are skipped,
as are citations, cross-references, formulas, code and file paths.
"""
if __package__ in (None, ""):  # allow `python src/check_report.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import re
from collections import defaultdict

from src.data import ROOT

REPORT_DIR = ROOT / "report"
GENERATED = REPORT_DIR / "generated"
REGISTER = REPORT_DIR / "typed_numbers.csv"

PLACEHOLDER = re.compile(r"\{\{([\w.-]+)\}\}")
NOT_NUMBERS = ("references",)                      # {{references}} is the bibliography, not a number
SKIP = (
    r"\{\{[^}]*\}\}",                              # placeholders and includes (checked separately)
    r"\[@[^\]]*\]",                                # citations
    r"@(?:fig|tbl|sec):[\w-]+",                    # cross-references
    r"\{[#.][^}]*\}",                              # labels and attributes of headings and figures
    r"\$\$.*?\$\$|\$[^$\n]*\$",                    # formulas
    r"`[^`]*`",                                    # code
    r"\]\([^)]*\)",                                # link and image targets
    r"\b[A-Za-z][A-Za-z]*-\d[\w.+]*",              # Sentinel-1, ResNet-50, Prithvi-100M, EO-2.0
    r"\b[A-Za-z]+\d\w*\+?",                        # R01, RQ3, E1, F1, MobileNetV2, Sen1Floods11, DeepLabv3+
)
NUMBER = re.compile(r"(?<![\w.])−?\d+(?:,\d{3})*(?:\.\d+)?%?")


def source_files():
    """The Markdown files of the report: report.md and every file it includes."""
    main = REPORT_DIR / "report.md"
    included = re.findall(r"\{\{include:([\w./-]+)\}\}", main.read_text())
    return [main] + [REPORT_DIR / name for name in included]


def placeholders(files):
    """{key: [file:line, ...]} for every {{key}} in the report source."""
    used = defaultdict(list)
    for path in files:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for key in PLACEHOLDER.findall(line):
                if key not in NOT_NUMBERS:
                    used[key].append(f"{path.name}:{number}")
    return used


def typed_numbers(text):
    """The numbers written by hand in a piece of report text, in order of appearance."""
    for pattern in SKIP:
        text = re.sub(pattern, " ", text, flags=re.S)
    text = re.sub(r"^\s*\d+\.\s", " ", text, flags=re.M)        # the markers of numbered lists
    return NUMBER.findall(text)


def typed_in_report(files):
    """{number: [file:line, ...]} for every typed number in the report source (front matter excluded)."""
    found = defaultdict(list)
    for path in files:
        body = re.sub(r"\A---\n.*?\n---\n", lambda m: "\n" * m.group(0).count("\n"), path.read_text(), flags=re.S)
        for number, line in enumerate(body.splitlines(), start=1):
            for token in typed_numbers(line):
                found[token].append(f"{path.name}:{number}")
    return found


def read_register():
    with open(REGISTER, newline="", encoding="utf-8") as f:
        return {row["number"]: row for row in csv.DictReader(f)}


def check_generated(files):
    """Problems with the generated numbers (an empty list means all of them are traced)."""
    numbers = json.loads((GENERATED / "numbers.json").read_text())
    sources = json.loads((GENERATED / "sources.json").read_text())
    problems = []
    for key, where in placeholders(files).items():
        if key not in numbers:
            problems.append(f"{{{{{key}}}}} ({where[0]}) has no value in report/generated/numbers.json")
        elif not sources.get(key):
            problems.append(f"{{{{{key}}}}} ({where[0]}) has no source in report/generated/sources.json")
        else:
            named = sources[key].split(":")[0].strip()
            if "/" in named and not (ROOT / named).exists():
                problems.append(f"{{{{{key}}}}}: its source file {named} does not exist")
    return problems


def check_typed(files):
    """Problems with the typed numbers: typed but not registered, or registered but no longer typed."""
    found, register = typed_in_report(files), read_register()
    problems = [f"{number} ({', '.join(where[:3])}) is typed in the report but not in report/typed_numbers.csv"
                for number, where in found.items() if number not in register]
    problems += [f"{number} is in report/typed_numbers.csv but no longer typed in the report"
                 for number in register if number not in found]
    problems += [f"{number} is registered without a note on where it was checked"
                 for number, row in register.items() if not row["checked_against"].strip()]
    return problems


def main():
    parser = argparse.ArgumentParser(description="Check that every number in the report can be traced.")
    parser.add_argument("--list", action="store_true", help="print every typed number and where it appears")
    args = parser.parse_args()

    files = source_files()
    used, typed = placeholders(files), typed_in_report(files)
    if args.list:
        register = read_register()
        for number in sorted(typed, key=lambda t: (float(re.sub(r"[,%]", "", t).replace("−", "-")), t)):
            meaning = register.get(number, {}).get("meaning", "NOT REGISTERED")
            print(f"{number:>8}  {meaning}  [{', '.join(typed[number][:4])}{', ...' if len(typed[number]) > 4 else ''}]")
        print()
    problems = check_generated(files) + check_typed(files)
    for problem in problems:
        print("UNTRACED:", problem)
    print(f"{len(used)} generated numbers (placeholders) and {len(typed)} typed numbers in {len(files)} files: "
          + ("all traced" if not problems else f"{len(problems)} problem(s)"))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
