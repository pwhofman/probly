#!/usr/bin/env python3
"""Render the rebuttal markdown table: AL methods (rows) x strategies (columns).

Reads the wandb cache written by ``inspect_al_runs.py`` and prints, per method,
the mean +/- std of a metric (default NAUC) over seeds for the four acquisition
strategies: TU (uncertainty:total), EU (uncertainty:epistemic), margin, random.

Usage::

    # Pre-req (any wandb-authenticated machine; writes scripts/al_analysis_out/):
    uv run python scripts/inspect_al_runs.py --refresh

    uv run python scripts/al_rebuttal_table.py --source probly/crv-al-cifar10
    uv run python scripts/al_rebuttal_table.py --metric final_accuracy

A method whose decomposition exposes fewer notions than a column asks for (see
``CIFAR10_METHOD_NOTIONS`` in ``run_al_sweep.py``) is filled from the notion it does
expose, and the cell carries a mark such as ``(AU)``.
"""
# ruff: noqa: T201

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd

# Sibling script; resolvable because the interpreter puts scripts/ first on sys.path.
from run_al_sweep import CIFAR10_METHOD_NOTIONS, DATASET_PROJECTS, WANDB_ENTITY

DEFAULT_INPUT = "scripts/al_analysis_out/wandb_cache_runs.pkl"

# (column header, strategy, notion) -- notion None means "not an uncertainty run".
COLUMNS: tuple[tuple[str, str, str | None], ...] = (
    ("TU", "uncertainty", "total"),
    ("EU", "uncertainty", "epistemic"),
    ("Margin", "margin", None),
    ("Random", "random", None),
)

# Legacy notion spellings (pre-#462) found in old wandb runs.
_NOTION_ALIASES = {"EU": "epistemic", "TU": "total", "AU": "aleatoric"}

# Display names (from the method configs' ``label`` fields); fallback is the raw
# method_label. Row order of the table follows this dict, then anything else.
PRETTY = {
    "base": "Base",
    "base+label_smoothing": "Base + label smoothing",
    "base+label_relaxation": "Base + label relaxation",
    "base+temperature_scaling": "Base + temp. scaling",
    "base+vector_scaling": "Base + vector scaling",
    "dropout": "Dropout",
    "dropconnect": "DropConnect",
    "bayesian": "BNN",
    "laplace": "Laplace",
    "vbll": "VBLL",
    "ensemble": "Deep Ensemble",
    "batchensemble": "BatchEnsemble",
    "subensemble": "SubEnsemble",
    "dare": "DARE",
    "duq": "DUQ",
    "ddu": "DDU",
    "sngp": "SNGP",
    "evidential_classification": "EDL",
    "posterior_network": "PostNet",
    "het_net": "HetNet",
    "credal_wrapper": "CreWra",
    "credal_ensembling": "CreEns",
    "credal_bnn": "CreBNN",
    "credal_relative_likelihood": "CreRL",
    "efficient_credal_prediction": "EffCre",
}

# Which run fills a notion column: the column's own notion, else the method's exposed
# notion in this order, marked with its abbreviation.
_COLUMN_PREFERENCE = {
    "total": ("total", "aleatoric", "epistemic"),
    "epistemic": ("epistemic", "aleatoric", "total"),
}
_MARKS = {"aleatoric": "AU", "epistemic": "EU", "total": "TU"}


def cell_notion(method: str, column: str) -> tuple[str, str | None]:
    """Return the notion whose run fills ``column`` for ``method`` and the mark to print, if any."""
    exposed = CIFAR10_METHOD_NOTIONS.get(method, ("total", "epistemic"))
    for notion in _COLUMN_PREFERENCE[column]:
        if notion in exposed:
            return notion, (None if notion == column else _MARKS[notion])
    return column, None


def effective_source(dataset: str, source: str | None) -> str | None:
    """The wandb project to read: the given one, else the dataset's campaign project, else every source."""
    if source is None and dataset in DATASET_PROJECTS:
        return f"{WANDB_ENTITY}/{DATASET_PROJECTS[dataset]}"
    return source


def load_runs(path: Path, dataset: str, source: str | None = None) -> pd.DataFrame:
    """Load finished runs of ``dataset`` from the cache, from the effective source project only.

    Without ``source`` a dataset with a campaign project in ``DATASET_PROJECTS`` reads that project
    alone, so pilot runs from other projects never fill a cell.
    """
    source = effective_source(dataset, source)
    df = cast("pd.DataFrame", pd.read_pickle(path))  # noqa: S301
    required = {"method", "dataset", "strategy", "seed", "state"}
    missing = required - set(df.columns)
    if missing:
        msg = f"{path} is missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    if "method_label" not in df.columns:
        df["method_label"] = df["method"]
    if "notion" in df.columns:
        df["notion"] = df["notion"].map(lambda n: _NOTION_ALIASES.get(n, n))
    df = df[(df["state"] == "finished") & (df["dataset"] == dataset)]
    if source is not None and "source" in df.columns:
        df = df[df["source"] == source]
    return df.reset_index(drop=True)


def format_cell(values: pd.Series, decimals: int, expected_seeds: int) -> str:
    """Format ``mean +/- std`` over seeds, flagging incomplete cells with (n=...)."""
    values = values.dropna()
    if values.empty:
        return "-"
    cell = f"{values.mean():.{decimals}f}"
    if len(values) > 1:
        cell += f" ± {values.std(ddof=1):.{decimals}f}"
    if len(values) != expected_seeds:
        cell += f" (n={len(values)})"
    return cell


def build_table(df: pd.DataFrame, metric: str, decimals: int, expected_seeds: int) -> str:
    """Return the markdown table string."""
    known = list(PRETTY)
    labels = sorted(
        df["method_label"].unique(),
        key=lambda m: (m not in known, known.index(m) if m in known else 0, str(m)),
    )
    lines = [
        "| Method | " + " | ".join(h for h, _, _ in COLUMNS) + " |",
        "|---|" + "---|" * len(COLUMNS),
    ]
    for label in labels:
        sub = df[df["method_label"] == label]
        method = str(sub["method"].iloc[0])
        cells = []
        for _, strategy, notion in COLUMNS:
            mask = sub["strategy"] == strategy
            mark = None
            if notion is not None:
                read_notion, mark = cell_notion(method, notion)
                mask &= sub["notion"] == read_notion
            # Keep one value per seed; a re-run seed would otherwise average twice.
            per_seed = sub[mask].sort_values("created_at").groupby("seed")[metric].last()
            cell = format_cell(per_seed, decimals, expected_seeds)
            cells.append(f"{cell} ({mark})" if mark and cell != "-" else cell)
        lines.append(f"| {PRETTY.get(str(label), str(label))} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    """Parse arguments, load the cache, and print the markdown table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", default=DEFAULT_INPUT, metavar="PATH", help=f"inspect_al_runs cache (default: {DEFAULT_INPUT})"
    )
    p.add_argument("--dataset", default="cifar10", help="dataset key to tabulate (default: cifar10)")
    p.add_argument("--metric", default="nauc", choices=("nauc", "final_accuracy"))
    p.add_argument("--decimals", type=int, default=3)
    p.add_argument("--expected-seeds", type=int, default=5, help="seed count a complete cell should have")
    p.add_argument("--source", default=None, metavar="ENTITY/PROJECT", help="only runs from this wandb project")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        p.error(f"{path} not found. Run `uv run python scripts/inspect_al_runs.py --refresh` first.")
    df = load_runs(path, args.dataset, source=args.source)
    if df.empty:
        p.error(f"No finished {args.dataset} runs in {path}. Refresh the cache?")

    print(build_table(df, args.metric, args.decimals, args.expected_seeds))
    n_runs = len(df)
    seeds = sorted(int(s) for s in df["seed"].dropna().unique())
    where = effective_source(args.dataset, args.source) or "all sources"
    print(
        f"\n{args.metric} mean ± std over seeds {seeds} "
        f"({n_runs} finished {args.dataset} runs from {where}; '-' = not run)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
