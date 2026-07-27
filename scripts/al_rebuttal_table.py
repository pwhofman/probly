#!/usr/bin/env python3
"""Render the rebuttal markdown table: AL methods (rows) x strategies (columns).

Reads the wandb cache written by ``inspect_al_runs.py`` and prints, per method,
the mean +/- std of a metric (default NAUC) over seeds for the four acquisition
strategies: TU (uncertainty:total), EU (uncertainty:epistemic), margin, random.

Usage::

    # Pre-req (any wandb-authenticated machine; writes scripts/al_analysis_out/):
    uv run python scripts/inspect_al_runs.py --refresh

    uv run python scripts/al_rebuttal_table.py
    uv run python scripts/al_rebuttal_table.py --metric final_accuracy
"""
# ruff: noqa: T201

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd

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
    "dropout": "Dropout",
    "evidential_classification": "Evidential",
    "vbll": "VBLL",
    "ensemble": "Deep Ensemble",
    "subensemble": "SubEnsemble",
}


def load_runs(path: Path, dataset: str) -> pd.DataFrame:
    """Load finished runs of ``dataset`` from the inspect_al_runs cache."""
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
        cells = []
        for _, strategy, notion in COLUMNS:
            mask = sub["strategy"] == strategy
            if notion is not None:
                mask &= sub["notion"] == notion
            # Keep one value per seed; a re-run seed would otherwise average twice.
            per_seed = sub[mask].sort_values("created_at").groupby("seed")[metric].last()
            cells.append(format_cell(per_seed, decimals, expected_seeds))
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
    p.add_argument("--expected-seeds", type=int, default=3, help="seed count a complete cell should have")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        p.error(f"{path} not found. Run `uv run python scripts/inspect_al_runs.py --refresh` first.")
    df = load_runs(path, args.dataset)
    if df.empty:
        p.error(f"No finished {args.dataset} runs in {path}. Refresh the cache?")

    print(build_table(df, args.metric, args.decimals, args.expected_seeds))
    n_runs = len(df)
    seeds = sorted(int(s) for s in df["seed"].dropna().unique())
    print(f"\n{args.metric} mean ± std over seeds {seeds} ({n_runs} finished {args.dataset} runs; '-' = not run).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
