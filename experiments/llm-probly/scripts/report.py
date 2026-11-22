"""The benchmark report (E4): every run, every method, with cost, in Markdown and HTML.

Run from ``experiments/llm-probly``::

    uv run python -m cli report                 # results/report.md and results/report.html
    uv run python -m cli report --n-boot 200    # faster, for a quick look

It reads every ``results/<model>/<dataset>_<n>/`` that has a data cache, scores all
method files there against the cache's labels and writes one report with:

1. the runs (model, dataset, questions, accuracy, precision and device),
2. methods x models on TriviaQA 200 (AUROC, best per model in bold) next to each
   method's family, access and cost from the registry,
3. one table per run: AUROC with its 95 % interval, the paired comparison with the
   best method of the run (best / tied / worse), AURC, PRR, ECE, measured seconds per
   question and the registry cost,
4. per run the cheapest method that is tied with the best one: the pick a user with
   that model would make.

Nothing is generated or computed with a model; the report only reads results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

from common.data import cache_path, load_jsonl
from common.io import RESULTS_ROOT
from common.registry import REGISTRY
from common.summary import MODEL_ORDER, summary_table
from scripts.run_benchmark import add_registry_columns

if TYPE_CHECKING:
    import pandas as pd

DATASET_ORDER = ("triviaqa", "mmlu")
REFERENCE = "dummy_random"  # random scores: a yardstick, never a pick
NO_SIGNAL = "none (no method beats random)"
NOT_TIED = "none tied with the best"  # e.g. only a white-box method reaches the best AUROC
API_ACCESS = ("text_only", "logprobs")  # what an OpenAI-compatible API gives
COMPARE_RUN = ("triviaqa", 200)  # the run every model has; used for the methods x models table


@dataclass
class RunReport:
    """Evaluated methods of one run.

    Attributes:
        model: Short model name (folder name).
        dataset: Dataset name.
        n: Number of questions.
        accuracy: Share of correct greedy answers.
        settings: Precision and device of the cache, e.g. ``"float32, cpu"``.
        table: One row per evaluated method (summary columns plus family, access, cost).
        pick: Cheapest registered method that is best or tied with the best (any access).
        pick_api: The same among methods that work through an API (text or log-probabilities).
    """

    model: str
    dataset: str
    n: int
    accuracy: float
    settings: str
    table: pd.DataFrame
    pick: str
    pick_api: str

    @property
    def name(self) -> str:
        """``<model>/<dataset>_<n>``, the results folder."""
        return f"{self.model}/{self.dataset}_{self.n}"


def model_key(model: str) -> tuple[int, str]:
    """Sort key: the Qwen sizes in order, other families after them."""
    return (MODEL_ORDER.index(model), model) if model in MODEL_ORDER else (len(MODEL_ORDER), model)


def find_runs(results_root: Path = RESULTS_ROOT, min_n: int = 100) -> list[tuple[str, str, int]]:
    """Every ``(model, dataset, n)`` with method results and a data cache, ``n >= min_n``.

    Smoke-test runs (5 questions) are skipped by ``min_n``.
    """
    runs = []
    for folder in results_root.glob("*/*_*"):
        dataset, _, n = folder.name.rpartition("_")
        if not n.isdigit() or int(n) < min_n or dataset not in DATASET_ORDER:
            continue
        model = folder.parent.name
        has_results = any(
            p.name != "summary.md" and not p.name.startswith("sample_count") for p in folder.glob("*.json")
        )
        if has_results and cache_path(model, int(n), dataset).exists():
            runs.append((model, dataset, int(n)))
    return sorted(runs, key=lambda r: (DATASET_ORDER.index(r[1]), r[2], model_key(r[0])))


def pick_method(table: pd.DataFrame, access: tuple[str, ...] | None = None) -> str:
    """Cheapest registered method among the best and the tied ones; higher AUROC breaks cost ties.

    Same rule as the selector (D1): the data cannot tell the tied methods apart from the
    best one, so the cheapest of them is the sensible choice. The random reference is
    never picked, and if it is itself tied with the best method, nothing is better than
    random on this run and :data:`NO_SIGNAL` is returned.

    Args:
        table: Evaluated methods of one run (``vs_best`` and ``auroc`` columns).
        access: Allowed access levels, e.g. :data:`API_ACCESS`; None allows all.

    Returns:
        The method name, :data:`NO_SIGNAL`, or :data:`NOT_TIED` if no allowed method is tied with the best.
    """
    reference = table[table["method"] == REFERENCE]
    if not reference.empty and reference["vs_best"].iloc[0] in {"best", "tied"}:
        return NO_SIGNAL
    allowed = [m for m in REGISTRY if m != REFERENCE and (access is None or REGISTRY[m].access in access)]
    candidates = table[table["vs_best"].isin(["best", "tied"]) & table["method"].isin(allowed)]
    if candidates.empty:
        return NOT_TIED
    candidates = candidates.assign(cost_value=[REGISTRY[m].cost for m in candidates["method"]])
    return str(candidates.sort_values(["cost_value", "auroc"], ascending=[True, False]).iloc[0]["method"])


def evaluate_run(model: str, dataset: str, n: int, results_root: Path = RESULTS_ROOT, n_boot: int = 1000) -> RunReport:
    """Score every method file of one run and pick the cheapest method tied with the best."""
    path = cache_path(model, n, dataset)
    rows = load_jsonl(path)
    meta_path = cache_path(model, n, dataset, suffix=".meta.json")
    settings = "?"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        settings = f"{meta['settings']['precision']}, {meta['environment']['device']}"
    table = summary_table(rows, run=f"{dataset}_{n}", methods=(), results_root=results_root / model, n_boot=n_boot)
    table = add_registry_columns(table[table["status"] == "ok"].reset_index(drop=True))
    table = table.sort_values("auroc", ascending=False).reset_index(drop=True)
    accuracy = sum(bool(row["correct"]) for row in rows) / len(rows)
    return RunReport(model, dataset, n, accuracy, settings, table, pick_method(table), pick_method(table, API_ACCESS))


def number(value: object, digits: int = 3) -> str:
    """A table cell: floats rounded, NaN and None empty."""
    if value is None or (isinstance(value, float) and value != value):  # noqa: PLR0124 - NaN check
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def run_rows(run: RunReport) -> tuple[list[str], list[list[str]]]:
    """Header and cells of one run's table (shared by Markdown and HTML)."""
    header = ["method", "family", "access", "cost", "AUROC", "95 % interval", "vs best", "Δ AUROC [paired 95 %]"]
    header += ["AURC", "PRR", "ECE", "s / question (method)"]
    cells = []
    for _, row in run.table.iterrows():
        delta = (
            ""
            if row["vs_best"] == "best"
            else f"{row['d_auroc']:+.3f} [{row['d_ci_low']:+.3f}, {row['d_ci_high']:+.3f}]"
        )
        name = f"{row['method']} ★" if row["method"] == run.pick else row["method"]
        cells.append(
            [
                name,
                row["family"],
                row["access"],
                row["cost"],
                number(row["auroc"]),
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
                row["vs_best"],
                delta,
                number(row["aurc"]),
                number(row["prr"]),
                number(row["ece"]),
                number(row["seconds_per_question"], 2),
            ]
        )
    return header, cells


def overview_rows(runs: list[RunReport]) -> tuple[list[str], list[list[str]]]:
    """Header and cells of the runs overview."""
    header = [
        "model",
        "dataset",
        "questions",
        "accuracy",
        "precision, device",
        "methods",
        "pick",
        "pick via API",
    ]
    cells = [
        [r.model, r.dataset, str(r.n), f"{r.accuracy:.3f}", r.settings, str(len(r.table)), r.pick, r.pick_api]
        for r in runs
    ]
    return header, cells


def compare_rows(runs: list[RunReport]) -> tuple[list[str], list[list[str]], set[tuple[int, int]]]:
    """Methods x models AUROC on the comparison run; also returns the (row, column) cells to highlight."""
    dataset, n = COMPARE_RUN
    chosen = sorted([r for r in runs if (r.dataset, r.n) == (dataset, n)], key=lambda r: model_key(r.model))
    methods = [m for m in REGISTRY if any(m in set(r.table["method"]) for r in chosen)]
    header = ["method", "family", "access", "cost", *[r.model for r in chosen]]
    cells, bold = [], set()
    for i, method in enumerate(methods):
        spec = REGISTRY[method]
        line = [method, spec.family, spec.access, f"{spec.cost:g}"]
        for j, run in enumerate(chosen):
            found = run.table[run.table["method"] == method]
            if found.empty:
                line.append("")
                continue
            line.append(number(float(found["auroc"].iloc[0])))
            if found["vs_best"].iloc[0] == "best":
                bold.add((i, 4 + j))
        cells.append(line)
    return header, cells, bold


def markdown_table(header: list[str], cells: list[list[str]], bold: set[tuple[int, int]] | None = None) -> list[str]:
    """Rows of a Markdown table."""
    bold = bold or set()
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for i, row in enumerate(cells):
        lines.append("| " + " | ".join(f"**{c}**" if (i, j) in bold and c else c for j, c in enumerate(row)) + " |")
    return lines


INTRO = [
    "AUROC: how well the uncertainty separates wrong from right answers (0.5 = random, 1 = perfect).",
    (
        "**vs best** compares every method with the run's best one by a paired bootstrap: *tied* means the "
        "95 % interval of the AUROC difference contains 0, so the data cannot tell the two apart. "
        "**cost** is generation passes per question (1 = the greedy answer alone; registry, CONTRACT.md). "
        "★ marks the pick: the cheapest method that is best or tied with the best. *Pick via API* only "
        "allows methods that need text or log-probabilities, which an OpenAI-compatible API returns. "
        "If the random reference is tied with the best method, no method beats random on that run."
    ),
]


def to_markdown(runs: list[RunReport], n_boot: int) -> str:
    """The whole report as Markdown."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# LLM-probly benchmark report",
        "",
        f"*Generated {stamp} by `python -m cli report` ({n_boot} bootstrap rounds).*",
        "",
    ]
    lines += [*INTRO, "", "## Runs", "", *markdown_table(*overview_rows(runs)), ""]
    header, cells, bold = compare_rows(runs)
    if cells:
        lines += [f"## Methods x models, {COMPARE_RUN[0]} {COMPARE_RUN[1]} (AUROC, best per model in bold)", ""]
        lines += [*markdown_table(header, cells, bold), ""]
    for run in runs:
        lines += [f"## {run.name}", "", f"Accuracy {run.accuracy:.3f}, {run.settings}.", ""]
        lines += [*markdown_table(*run_rows(run)), ""]
    return "\n".join(lines)


CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; padding: 0 1rem;
       color: #1d1d1f; line-height: 1.45; }
h1 { margin-bottom: 0.2rem; } h2 { margin-top: 2.2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.2rem; }
table { border-collapse: collapse; margin: 0.6rem 0 1rem; font-size: 0.9rem; }
th, td { padding: 0.3rem 0.6rem; border-bottom: 1px solid #eee; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; } th { background: #f5f5f7; }
tr.best td { background: #e7f6ec; } tr.tied td { background: #f3faf5; } tr.pick td:first-child { font-weight: 600; }
td.bold { font-weight: 700; } .note { color: #555; font-size: 0.9rem; }
"""


def html_table(
    header: list[str],
    cells: list[list[str]],
    row_classes: list[str] | None = None,
    bold: set[tuple[int, int]] | None = None,
) -> str:
    """An HTML table with escaped cells."""
    bold = bold or set()
    row_classes = row_classes or [""] * len(cells)
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = []
    for i, (row, cls) in enumerate(zip(cells, row_classes, strict=True)):
        tds = "".join(
            f'<td class="bold">{html.escape(c)}</td>' if (i, j) in bold else f"<td>{html.escape(c)}</td>"
            for j, c in enumerate(row)
        )
        body.append(f'<tr class="{cls}">{tds}</tr>')
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def row_class(vs_best: str, is_pick: bool) -> str:
    """CSS classes of a run-table row: ``best`` / ``tied`` shading, ``pick`` for the pick."""
    classes = [vs_best] if vs_best in {"best", "tied"} else []
    if is_pick:
        classes.append("pick")
    return " ".join(classes)


def to_html(runs: list[RunReport], n_boot: int) -> str:
    """The whole report as one self-contained HTML page."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8"><title>LLM-probly benchmark report</title>',
        f"<style>{CSS}</style></head><body>",
        "<h1>LLM-probly benchmark report</h1>",
        f'<p class="note">Generated {stamp} by <code>python -m cli report</code> ({n_boot} bootstrap rounds).</p>',
        *(f'<p class="note">{html.escape(p).replace("**", "")}</p>' for p in INTRO),
        "<h2>Runs</h2>",
        html_table(*overview_rows(runs)),
    ]
    header, cells, bold = compare_rows(runs)
    if cells:
        parts += [
            f"<h2>Methods x models, {COMPARE_RUN[0]} {COMPARE_RUN[1]} (AUROC, best per model in bold)</h2>",
            html_table(header, cells, bold=bold),
        ]
    for run in runs:
        header, cells = run_rows(run)
        classes = [row_class(str(row["vs_best"]), row["method"] == run.pick) for _, row in run.table.iterrows()]
        parts += [
            f"<h2>{html.escape(run.name)}</h2>",
            f'<p class="note">Accuracy {run.accuracy:.3f}, {html.escape(run.settings)}.</p>',
            html_table(header, cells, classes),
        ]
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def main() -> None:
    """Write results/report.md and results/report.html."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap rounds for the intervals")
    parser.add_argument("--min-n", type=int, default=100, help="skip runs with fewer questions (smoke tests)")
    parser.add_argument("--out", default=str(RESULTS_ROOT / "report"), help="path without extension")
    args = parser.parse_args()

    found = find_runs(min_n=args.min_n)
    runs = []
    for model, dataset, n in found:
        print(f"evaluating {model}/{dataset}_{n} ...", flush=True)
        runs.append(evaluate_run(model, dataset, n, n_boot=args.n_boot))
    for suffix, text in ((".md", to_markdown(runs, args.n_boot)), (".html", to_html(runs, args.n_boot))):
        path = Path(f"{args.out}{suffix}")
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
