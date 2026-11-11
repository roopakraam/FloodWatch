"""Build the report: report/report.md -> report/report.html -> report/FloodWatch_report.pdf.

The Markdown source contains no hand-typed results. Numbers and tables are pulled in at build time:

    {{key}}             a value from report/generated/numbers.json   (written by src/report_assets.py from results/)
    {{table:name}}      report/generated/tables/<name>.md             (same script)
    [@key] [@a; @b]     numbered citations, reference list from report/references.bib at {{references}}
    ![caption](path){#fig:id width=80%}      a numbered figure;  @fig:id  ->  "Figure 3"
    Table: caption {#tbl:id}                 caption line right above a table;  @tbl:id  ->  "Table 2"
    ## Heading {#sec:id}                     numbered section;  @sec:id  ->  "Section 4"
    $...$ and $$...$$                        LaTeX maths, converted to MathML

A missing key, table, citation or cross-reference stops the build, so the PDF cannot silently go stale.

The PDF is printed by a Chromium-based browser in headless mode (Chrome or Edge; on WSL the Windows
installation is used). Set CHROME_PATH to point at another browser binary.

Run from the project root:
    python -m src.report_assets      # refresh numbers and tables from results/
    python -m src.build_report       # HTML + PDF      (--html-only to skip the PDF)
"""
if __package__ in (None, ""):  # allow `python src/build_report.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from src.data import ROOT

REPORT_DIR = ROOT / "report"
SOURCE = REPORT_DIR / "report.md"
GENERATED = REPORT_DIR / "generated"
HTML_OUT = REPORT_DIR / "report.html"
PDF_OUT = REPORT_DIR / "FloodWatch_report.pdf"

CHROME_CANDIDATES = (
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge",
)


# ---- bibliography -----------------------------------------------------------------------------------------

def parse_bibtex(text):
    """Minimal BibTeX reader: {key: {"type": ..., field: value}}. Handles nested braces in values."""
    entries = {}
    for match in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        kind, key = match.group(1).lower(), match.group(2)
        depth, i = 1, match.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body, fields, pos = text[match.end():i - 1], {"type": kind}, 0
        while True:
            m = re.compile(r"\s*(\w+)\s*=\s*").match(body, pos)
            if not m:
                break
            j = m.end()
            if body[j] == "{":
                depth, k = 1, j + 1
                while depth:
                    depth += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                value, pos = body[j + 1:k - 1], k
            else:
                k = body.find(",", j)
                k = len(body) if k == -1 else k
                value, pos = body[j:k].strip().strip('"'), k
            fields[m.group(1).lower()] = value
            pos = body.find(",", pos) + 1 if "," in body[pos:] else len(body)
        entries[key] = fields
    return entries


def detex(s):
    """LaTeX markup used in the .bib file -> plain text."""
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    for a, b in ((r'{\"o}', "ö"), (r"{\'o}", "ó"), (r'\"o', "ö"), (r"\'o", "ó"), ("--", "–"), (r"\&", "&")):
        s = s.replace(a, b)
    return re.sub(r"[{}]", "", s).strip()


def format_authors(raw):
    names = [detex(n) for n in re.split(r"\s+and\s+", raw.strip())]
    out = []
    for name in names:
        if name == "others":
            out.append("et al.")
        elif "," in name:
            family, given = [p.strip() for p in name.split(",", 1)]
            initials = " ".join("-".join(part[0] + "." for part in word.split("-")) for word in given.split())
            out.append(f"{initials} {family}")
        else:
            out.append(name)
    if out[-1] == "et al.":
        return ", ".join(out[:-1]) + " et al"          # the caller adds the final full stop
    if len(out) > 2:
        return ", ".join(out[:-1]) + ", and " + out[-1]
    return " and ".join(out)


def format_reference(e):
    """One reference as an HTML string (numeric style: authors, title, venue, volume, pages, year, DOI)."""
    parts = [format_authors(e["author"]) + ".", f"{html.escape(detex(e['title']))}."]
    venue = e.get("journal") or e.get("booktitle") or e.get("institution")
    if venue:
        v = f"<em>{html.escape(detex(venue))}</em>"
        if e.get("volume"):
            v += f", {e['volume']}" + (f"({e['number']})" if e.get("number") else "")
        if e.get("pages"):
            v += f":{detex(e['pages'])}" if e.get("volume") else f", pp. {detex(e['pages'])}"
        parts.append(v + f", {e['year']}.")
    else:
        how = detex(e.get("howpublished", ""))
        how = re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', html.escape(how))
        parts.append((how + ", " if how else "") + f"{e['year']}.")
    if e.get("doi"):
        parts.append(f'doi:<a href="https://doi.org/{e["doi"]}">{e["doi"]}</a>')
    elif venue and "http" in e.get("howpublished", ""):
        url = re.search(r"https?://[^\s}]+", e["howpublished"]).group(0)
        parts.append(f'<a href="{url}">{url}</a>')
    return " ".join(parts)


# ---- Markdown pre-processing ----------------------------------------------------------------------------------

class Builder:
    def __init__(self, text):
        self.text = text
        self.numbers = json.loads((GENERATED / "numbers.json").read_text()) if (GENERATED / "numbers.json").exists() else {}
        self.bib = parse_bibtex((REPORT_DIR / "references.bib").read_text())
        self.cited, self.labels, self.used_keys = [], {}, set()

    def fail(self, message):
        raise SystemExit(f"report build failed: {message}")

    # -- front matter: "key: value" lines between two '---' lines at the top
    def front_matter(self):
        meta = {}
        m = re.match(r"---\n(.*?)\n---\n", self.text, flags=re.S)
        if m:
            for line in m.group(1).splitlines():
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
            self.text = self.text[m.end():]
        return meta

    def include_tables(self):
        def repl(m):
            path = GENERATED / "tables" / f"{m.group(1)}.md"
            if not path.exists():
                self.fail(f"generated table {m.group(1)!r} not found (run python -m src.report_assets)")
            return path.read_text().strip()
        self.text = re.sub(r"\{\{table:([\w-]+)\}\}", repl, self.text)

    def fill_numbers(self):
        def repl(m):
            key = m.group(1)
            if key == "references":
                return m.group(0)
            if key not in self.numbers:
                self.fail(f"number {key!r} is not in report/generated/numbers.json")
            self.used_keys.add(key)
            return str(self.numbers[key])
        self.text = re.sub(r"\{\{([\w.-]+)\}\}", repl, self.text)

    def maths(self):
        from latex2mathml.converter import convert

        def block(m):
            return f'\n\n<div class="math">{convert(m.group(1).strip(), display="block")}</div>\n\n'

        def inline(m):
            return convert(m.group(1).strip())
        self.text = re.sub(r"\$\$(.+?)\$\$", block, self.text, flags=re.S)
        self.text = re.sub(r"(?<![\\$\w])\$([^$\n]+?)\$(?![\w$])", inline, self.text)

    def number_things(self):
        """Number sections, figures and tables and remember their labels."""
        counters = {"fig": 0, "tbl": 0}
        sec = [0, 0]
        lines, out = self.text.split("\n"), []
        for line in lines:
            h = re.match(r"^(##|###)\s+(.*?)\s*(\{([^}]*)\})?\s*$", line)
            if h:
                level, title, attrs = len(h.group(1)), h.group(2), h.group(4) or ""
                ident = re.search(r"#([\w:-]+)", attrs)
                if ".unnumbered" in attrs:
                    number = None
                elif level == 2:
                    sec = [sec[0] + 1, 0]
                    number = f"{sec[0]}"
                else:
                    sec[1] += 1
                    number = f"{sec[0]}.{sec[1]}"
                if ident and number:
                    self.labels[ident.group(1)] = f"Section {number}"
                anchor = f' id="{ident.group(1)}"' if ident else ""
                label = f'<span class="secnum">{number}</span> ' if number else ""
                out.append(f"<h{level}{anchor}>{label}{title}</h{level}>")
                continue
            f = re.match(r"^!\[(.*)\]\(([^)]+)\)\{#(fig:[\w-]+)([^}]*)\}\s*$", line)
            if f:
                counters["fig"] += 1
                caption, path, ident, attrs = f.groups()
                self.labels[ident] = f"Figure {counters['fig']}"
                if not (REPORT_DIR / path).exists():
                    self.fail(f"figure file {path!r} does not exist")
                width = re.search(r"width=([\d.]+%)", attrs)
                style = f' style="width:{width.group(1)}"' if width else ""
                out.append(f'<figure id="{ident}"><img src="{path}"{style} alt="">'
                           f'<figcaption markdown="1"><b>Figure {counters["fig"]}.</b> {caption}</figcaption></figure>')
                continue
            t = re.match(r"^Table:\s+(.*?)\s*\{#(tbl:[\w-]+)\}\s*$", line)
            if t:
                counters["tbl"] += 1
                self.labels[t.group(2)] = f"Table {counters['tbl']}"
                out.append(f'<div class="tablecaption" id="{t.group(2)}"><b>Table {counters["tbl"]}.</b> {t.group(1)}</div>\n')
                continue
            out.append(line)
        self.text = "\n".join(out)

    def cross_references(self):
        def repl(m):
            if m.group(1) not in self.labels:
                self.fail(f"cross-reference @{m.group(1)} has no target")
            return f'<a class="xref" href="#{m.group(1)}">{self.labels[m.group(1)]}</a>'
        self.text = re.sub(r"(?<![\w\[;])@((?:fig|tbl|sec):[\w-]+)", repl, self.text)

    def citations(self):
        def repl(m):
            nums = []
            for key in re.findall(r"@([\w-]+)", m.group(0)):
                if key not in self.bib:
                    self.fail(f"citation @{key} is not in references.bib")
                if key not in self.cited:
                    self.cited.append(key)
                n = self.cited.index(key) + 1
                nums.append(f'<a class="cite" href="#ref-{key}">{n}</a>')
            return "[" + ", ".join(nums) + "]"
        self.text = re.sub(r"\[@[\w-]+(?:;\s*@[\w-]+)*\]", repl, self.text)
        refs = "\n".join(f'<li id="ref-{k}">{format_reference(self.bib[k])}</li>' for k in self.cited)
        if "{{references}}" not in self.text:
            self.fail("the report has no {{references}} placeholder")
        self.text = self.text.replace("{{references}}", f'<ol class="references">\n{refs}\n</ol>')

    def to_html(self):
        from markdown_it import MarkdownIt

        meta = self.front_matter()
        self.include_tables()
        self.fill_numbers()
        self.number_things()
        self.cross_references()
        self.citations()
        self.maths()
        md = MarkdownIt("commonmark", {"html": True, "typographer": True}).enable(["table", "replacements", "smartquotes"])
        body = md.render(self.text)
        # captions were written as raw HTML, so render the Markdown inside them
        body = re.sub(r'<figcaption markdown="1">(.*?)</figcaption>',
                      lambda m: "<figcaption>" + md.renderInline(m.group(1)) + "</figcaption>", body, flags=re.S)
        body = re.sub(r'(<div class="tablecaption"[^>]*>)(.*?)(</div>)',
                      lambda m: m.group(1) + md.renderInline(m.group(2)) + m.group(3), body, flags=re.S)
        return meta, body


def page(meta, body):
    css = (REPORT_DIR / "style.css").read_text()
    title = html.escape(meta.get("title", "Report"))
    block = [f'<h1 class="title">{title}</h1>']
    if meta.get("subtitle"):
        block.append(f'<p class="subtitle">{html.escape(meta["subtitle"])}</p>')
    byline = " · ".join(html.escape(meta[k]) for k in ("author", "affiliation", "date") if meta.get(k))
    if byline:
        block.append(f'<p class="byline">{byline}</p>')
    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>{title}</title>\n'
            f"<style>\n{css}\n</style>\n</head>\n<body>\n<header>\n" + "\n".join(block) + f"\n</header>\n{body}\n</body>\n</html>\n")


# ---- PDF ------------------------------------------------------------------------------------------------------

def find_browser():
    custom = os.environ.get("CHROME_PATH")
    for candidate in ([custom] if custom else []) + list(CHROME_CANDIDATES):
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return candidate
    return None


def print_pdf(html_path, pdf_path):
    browser = find_browser()
    if browser is None:
        raise SystemExit("no Chrome / Edge / Chromium found; set CHROME_PATH, or open report/report.html and print to PDF")
    if browser.endswith(".exe"):  # a Windows browser started from WSL needs Windows paths
        def win(p):
            return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()
        url, target = "file:///" + win(html_path).replace("\\", "/"), win(pdf_path)
    else:
        url, target = html_path.resolve().as_uri(), str(pdf_path)
    pdf_path.unlink(missing_ok=True)
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={target}", url],
                   capture_output=True, text=True, timeout=180)
    if not pdf_path.exists():
        raise SystemExit(f"the browser did not produce {pdf_path}")


def build(html_only=False):
    builder = Builder(SOURCE.read_text())
    meta, body = builder.to_html()
    HTML_OUT.write_text(page(meta, body))
    print(f"wrote {HTML_OUT.relative_to(ROOT)}: {len(builder.cited)} references, "
          f"{sum(v.startswith('Figure') for v in builder.labels.values())} figures, "
          f"{sum(v.startswith('Table') for v in builder.labels.values())} tables, "
          f"{len(builder.used_keys)} numbers filled from results")
    if not html_only:
        print_pdf(HTML_OUT, PDF_OUT)
        print(f"wrote {PDF_OUT.relative_to(ROOT)} ({PDF_OUT.stat().st_size / 1024:.0f} kB)")
    return builder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the FloodWatch report (HTML and PDF).")
    parser.add_argument("--html-only", action="store_true")
    build(parser.parse_args().html_only)
