#!/usr/bin/env python3
r"""Render the camera-ready active learning figure of one dataset (the paper's Figure 15 layout).

Top: NAUC per method for the TU and EU acquisition columns, with base + margin and base + random
as horizontal lines; hatched bars are fallback notions (see ``cell_notion`` in
``al_rebuttal_table.py``). Bottom: budget curves (test accuracy against labeled samples, mean over
seeds with a standard-deviation band) of the best methods per notion, with the two base baselines.
Reads the caches written by ``inspect_al_runs.py``.

Usage::

    uv run python scripts/al_figure.py --dataset cifar10 --source probly/crv-al-cifar10 \\
        --out scripts/al_analysis_out/al_cifar10.pdf
"""
# ruff: noqa: T201

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

# Sibling script; resolvable because the interpreter puts scripts/ first on sys.path.
from al_rebuttal_table import PRETTY, cell_notion, load_runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = "scripts/al_analysis_out"
NOTION_PANELS: tuple[tuple[str, str], ...] = (("TU", "total"), ("EU", "epistemic"), ("AU", "aleatoric"))
BASELINES: tuple[tuple[str, str, str], ...] = (("margin", "base + margin", "--"), ("random", "base + random", ":"))


def per_seed(sub: pd.DataFrame, column: str) -> pd.Series:
    """One value per seed; a re-run seed keeps its latest run."""
    return cast("pd.Series", sub.sort_values("created_at").groupby("seed")[column].last())


def is_base(label: str) -> bool:
    """Whether a method label is the base model or one of its variants."""
    return label == "base" or label.startswith("base+")


def column_runs(df: pd.DataFrame, label: str, notion: str) -> tuple[pd.DataFrame, str | None]:
    """Uncertainty runs filling the ``notion`` column for a method label, and the fallback mark."""
    sub = df[df["method_label"] == label]
    read_notion, mark = (notion, None) if notion == "aleatoric" else cell_notion(str(sub["method"].iloc[0]), notion)
    return sub[(sub["strategy"] == "uncertainty") & (sub["notion"] == read_notion)], mark


def base_runs(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Runs of the plain base model with the given acquisition strategy."""
    return df[(df["method_label"] == "base") & (df["strategy"] == strategy)]


def curve(df_hist: pd.DataFrame, runs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Mean and standard deviation of test accuracy per labeled-set size over the runs' seeds."""
    run_ids = per_seed(runs, "run_id").tolist()
    hist = df_hist[df_hist["run_id"].isin(run_ids)]
    if hist.empty:
        return None
    per_run = hist.groupby(["run_id", "labeled_size"])["test_accuracy"].last().reset_index()
    stats = per_run.groupby("labeled_size")["test_accuracy"].agg(["mean", "std"]).reset_index()
    return stats["labeled_size"].to_numpy(), stats["mean"].to_numpy(), stats["std"].fillna(0.0).to_numpy()


def ranking(df: pd.DataFrame, notion: str) -> list[tuple[str, float, str | None]]:
    """Method labels with their mean NAUC for the notion column, best first."""
    rows: list[tuple[str, float, str | None]] = []
    for label in sorted(df["method_label"].unique()):
        if is_base(label):
            continue
        runs, mark = column_runs(df, label, notion)
        if not runs.empty:
            rows.append((label, float(per_seed(runs, "nauc").mean()), mark))
    return sorted(rows, key=lambda r: r[1], reverse=True)


def plot_bars(ax: plt.Axes, df: pd.DataFrame) -> None:
    """NAUC bars per method for the TU and EU columns with the base baselines as lines."""
    tu = {label: (value, mark) for label, value, mark in ranking(df, "total")}
    eu = {label: (value, mark) for label, value, mark in ranking(df, "epistemic")}
    labels = sorted(
        set(tu) | set(eu), key=lambda m: max(tu.get(m, (0.0, None))[0], eu.get(m, (0.0, None))[0]), reverse=True
    )
    x = np.arange(len(labels))
    width = 0.38
    for offset, column, table, color in ((-width / 2, "TU", tu, "tab:red"), (width / 2, "EU", eu, "tab:blue")):
        values = [table.get(m, (np.nan, None))[0] for m in labels]
        hatches = ["//" if table.get(m, (np.nan, None))[1] else "" for m in labels]
        bars = ax.bar(x + offset, values, width, label=column, color=color, alpha=0.85)
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)
    for strategy, name, style in BASELINES:
        runs = base_runs(df, strategy)
        if not runs.empty:
            ax.axhline(float(per_seed(runs, "nauc").mean()), linestyle=style, color="black", linewidth=1, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(m, m) for m in labels], rotation=45, ha="right", fontsize=8)
    finite = [v for table in (tu, eu) for v, _ in table.values() if np.isfinite(v)]
    if finite:
        ax.set_ylim(max(0.0, min(finite) - 0.05), min(1.0, max(finite) + 0.03))
    ax.set_ylabel("NAUC")
    ax.set_title("NAUC per method (hatched: fallback notion)")
    ax.legend(fontsize=8, ncol=4)


def plot_curves(ax: plt.Axes, df: pd.DataFrame, df_hist: pd.DataFrame, column: str, notion: str, top: int) -> list[str]:
    """Budget curves of the ``top`` methods by the notion column plus the base baselines; returns the legend entries."""
    printed: list[str] = []
    for strategy, name, style in BASELINES:
        runs = base_runs(df, strategy)
        c = curve(df_hist, runs)
        if c is not None:
            ax.plot(
                c[0], c[1], linestyle=style, color="black", label=f"{name} (NAUC={per_seed(runs, 'nauc').mean():.3f})"
            )
    for label, value, mark in ranking(df, notion)[:top]:
        runs, _ = column_runs(df, label, notion)
        c = curve(df_hist, runs)
        if c is None:
            continue
        name = PRETTY.get(label, label) + (f" ({mark})" if mark else "")
        ax.plot(c[0], c[1], label=f"{name} (NAUC={value:.3f})")
        ax.fill_between(c[0], c[1] - c[2], c[1] + c[2], alpha=0.15)
        printed.append(f"{name} {value:.3f}")
    ax.set_xlabel("Labeled samples")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Best methods by {column}")
    ax.legend(fontsize=7)
    return printed


def render(df: pd.DataFrame, df_hist: pd.DataFrame, out: Path, top: int) -> None:
    """Lay out the bar panel above one curve panel per notion that has runs, and save the figure."""
    panels = [(column, notion) for column, notion in NOTION_PANELS if ranking(df, notion)]
    fig = plt.figure(figsize=(4.2 * max(len(panels), 2) + 1, 8))
    grid = fig.add_gridspec(2, max(len(panels), 1), height_ratios=[1.1, 1.0])
    plot_bars(fig.add_subplot(grid[0, :]), df)
    for i, (column, notion) in enumerate(panels):
        printed = plot_curves(fig.add_subplot(grid[1, i]), df, df_hist, column, notion, top)
        print(f"{column}: " + "; ".join(printed))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> int:
    """Parse arguments, load the caches, and render the figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--source", default=None, metavar="ENTITY/PROJECT", help="only runs from this wandb project")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="directory with the inspect_al_runs caches")
    p.add_argument("--out", required=True, metavar="PATH", help="output figure (pdf or png)")
    p.add_argument("--top", type=int, default=5, help="methods per budget-curve panel")
    args = p.parse_args()

    cache = Path(args.cache_dir)
    df = load_runs(cache / "wandb_cache_runs.pkl", args.dataset, source=args.source)
    if df.empty:
        p.error(f"no finished {args.dataset} runs in {cache}; refresh the cache first.")
    df_hist = cast("pd.DataFrame", pd.read_pickle(cache / "wandb_cache_history.pkl"))  # noqa: S301
    render(df, df_hist, Path(args.out), args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
