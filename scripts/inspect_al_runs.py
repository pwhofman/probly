#!/usr/bin/env python3
"""Inventory and visualize active-learning runs from wandb.

Pulls runs from one or more ``<entity>/<project>`` sources (defaults:
``speyewear/jakubpaplham-al`` and ``probly/max-test``), merges them, then writes:

- ``inventory.csv``         seed-pivoted state + NAUC table
- ``coverage_gaps.md``      missing/non-finished seeds per config
- ``learning_curves_<dataset>.png``  mean +/- std accuracy vs labeled_size
- ``nauc_<dataset>.png``    per-method NAUC bars (uncertainty vs margin/random)

Caches the wandb fetch as pickle so re-runs are fast. Pass ``--refresh``
to invalidate the cache. Pass ``--source entity/project`` (repeatable) to
override the default sources.
"""
# ruff: noqa: T201, ANN401, D103

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

DEFAULT_SOURCES: tuple[str, ...] = ("speyewear/jakubpaplham-al", "probly/max-test")
DEFAULT_OUT = "scripts/al_analysis_out"
HISTORY_KEYS = ("iteration", "labeled_size", "test_accuracy")
PROGRESS_EVERY = 25

logger = logging.getLogger(__name__)


def _to_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


# Values that mean "no override" for each tweak field; anything else is treated
# as part of the method identity for the purposes of plotting / inventorying.
_DEFAULT_TWEAKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("calibration", ("none",)),
    ("supervised_loss", ("cross_entropy",)),
    ("conformal", ("none",)),
)


def _method_label(method: Any, **tweaks: Any) -> str:
    """Compose a method label that absorbs non-default calibration / loss / conformal.

    Examples:
        ``base`` -> ``base``
        ``base`` + ``calibration=temperature_scaling`` -> ``base+temperature_scaling``
        ``base`` + ``supervised_loss=label_smoothing`` -> ``base+label_smoothing``
        ``dropout`` + ``calibration=temperature_scaling`` -> ``dropout+temperature_scaling``
    """
    if method is None or (isinstance(method, float) and pd.isna(method)):
        return ""
    suffixes: list[str] = []
    for field, defaults in _DEFAULT_TWEAKS:
        value = tweaks.get(field)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if str(value) in defaults:
            continue
        suffixes.append(str(value))
    return f"{method}+{'+'.join(suffixes)}" if suffixes else str(method)


def _derive_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``method_label`` (calibration/loss/conformal-aware) and rebuild ``config_key``.

    Uses ``method_label`` instead of raw ``method`` so that e.g.
    ``base+temperature_scaling|openml_6|margin`` is treated as a distinct config from
    plain ``base|openml_6|margin`` everywhere downstream.
    """
    if df.empty:
        return df
    df = df.copy()

    def _row(field: str, i: int) -> Any:
        if field not in df.columns:
            return None
        return df[field].iloc[i]

    df["method_label"] = [
        _method_label(
            df["method"].iloc[i],
            calibration=_row("calibration", i),
            supervised_loss=_row("supervised_loss", i),
            conformal=_row("conformal", i),
        )
        for i in range(len(df))
    ]

    notion_suffix = df["notion"].apply(
        lambda n: f":{n}" if n is not None and not (isinstance(n, float) and pd.isna(n)) else ""
    )
    df["config_key"] = (
        df["method_label"].astype(str)
        + "|"
        + df["dataset"].astype(str)
        + "|"
        + df["strategy"].astype(str)
        + notion_suffix
    )
    return df


def _flat_record(run: Any, source: str) -> dict[str, Any]:
    cfg = run.config or {}
    method = (cfg.get("method") or {}).get("name")

    dataset_cfg = cfg.get("dataset") or {}
    ds_name = dataset_cfg.get("name")
    openml_id = dataset_cfg.get("openml_id")
    dataset_full = f"openml_{openml_id}" if ds_name == "openml" and openml_id is not None else ds_name

    strat_cfg = cfg.get("al_strategy") or {}
    strategy = strat_cfg.get("name")
    notion = strat_cfg.get("notion") if strategy == "uncertainty" else None
    notion_suffix = f":{notion}" if notion else ""
    config_key = f"{method}|{dataset_full}|{strategy}{notion_suffix}"

    summary = run.summary or {}
    return {
        "run_id": run.id,
        "source": source,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "method": method,
        "dataset": dataset_full,
        "strategy": strategy,
        "notion": notion,
        "config_key": config_key,
        "seed": cfg.get("seed"),
        "initial_size": cfg.get("initial_size"),
        "query_size": cfg.get("query_size"),
        "n_iterations": cfg.get("n_iterations"),
        "calibration": (cfg.get("calibration") or {}).get("name"),
        "conformal": (cfg.get("conformal") or {}).get("name"),
        "supervised_loss": (cfg.get("supervised_loss") or {}).get("name"),
        "nauc": _to_float(summary.get("nauc")),
        "final_accuracy": _to_float(summary.get("final_accuracy")),
    }


def _is_al_run(rec: dict[str, Any]) -> bool:
    """Filter out non-AL runs (e.g. plain training runs) — they have no al_strategy."""
    return rec.get("strategy") is not None


def fetch_one_source(source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch every AL run + per-iteration history from a single ``entity/project``."""
    api = wandb.Api()
    runs = api.runs(source, per_page=500)

    records: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    skipped_non_al = 0
    skipped_errors = 0
    for i, run in enumerate(runs, start=1):
        try:
            rec = _flat_record(run, source=source)
        except Exception as e:  # noqa: BLE001
            run_id = getattr(run, "id", "?")
            logger.warning("_flat_record() failed for %s: %s", run_id, e)
            skipped_errors += 1
            continue
        if not _is_al_run(rec):
            skipped_non_al += 1
            continue
        records.append(rec)

        if rec["state"] == "finished":
            try:
                hist = run.history(keys=list(HISTORY_KEYS), pandas=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("history() failed for %s: %s", run.id, e)
                hist = None
            if hist is not None and len(hist):
                hist = hist.copy()
                hist["run_id"] = run.id
                history_frames.append(hist)

        if i % PROGRESS_EVERY == 0:
            print(f"  [{source}] fetched {i} runs...", flush=True)

    if skipped_non_al:
        print(f"  [{source}] skipped {skipped_non_al} non-AL runs (no al_strategy in config)")
    if skipped_errors:
        print(f"  [{source}] skipped {skipped_errors} runs due to wandb fetch errors (retry --refresh later)")

    df_runs = pd.DataFrame(records)
    df_hist = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame({k: [] for k in (*HISTORY_KEYS, "run_id")})
    )
    return df_runs, df_hist


def fetch_all(sources: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and concatenate AL runs from each source."""
    runs_frames: list[pd.DataFrame] = []
    hist_frames: list[pd.DataFrame] = []
    for source in sources:
        print(f"Fetching from wandb {source}...")
        df_runs_one, df_hist_one = fetch_one_source(source)
        print(f"  [{source}] {len(df_runs_one)} AL runs, {len(df_hist_one)} history rows")
        runs_frames.append(df_runs_one)
        hist_frames.append(df_hist_one)

    df_runs = pd.concat(runs_frames, ignore_index=True) if runs_frames else pd.DataFrame()
    df_hist = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    if "seed" in df_runs.columns:
        df_runs["seed"] = pd.to_numeric(df_runs["seed"], errors="coerce").astype("Int64")
    df_runs = _derive_method_labels(df_runs)
    return df_runs, df_hist


def load_or_fetch(sources: list[str], out_dir: Path, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs_path = out_dir / "wandb_cache_runs.pkl"
    hist_path = out_dir / "wandb_cache_history.pkl"
    if not refresh and runs_path.exists() and hist_path.exists():
        print(f"Loading cached runs from {runs_path}")
        df_runs_cached = cast("pd.DataFrame", pd.read_pickle(runs_path))  # noqa: S301
        df_hist_cached = cast("pd.DataFrame", pd.read_pickle(hist_path))  # noqa: S301
        return _derive_method_labels(df_runs_cached), df_hist_cached

    df_runs, df_hist = fetch_all(sources)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_runs.to_pickle(runs_path)
    df_hist.to_pickle(hist_path)
    return df_runs, df_hist


def _format_cell(state: str, nauc: float | None) -> str:
    if state == "finished" and nauc is not None:
        return f"finished ({nauc:.3f})"
    return str(state)


def write_inventory(df_runs: pd.DataFrame, out_dir: Path) -> None:
    if df_runs.empty:
        print("No runs found.")
        return
    work = df_runs.copy()
    work["cell"] = [_format_cell(s, n) for s, n in zip(work["state"], work["nauc"], strict=True)]
    pivot = work.pivot_table(
        index="config_key",
        columns="seed",
        values="cell",
        aggfunc=lambda xs: "; ".join(map(str, xs)),
    )
    pivot = pivot.sort_index(axis=1).sort_index(axis=0)
    csv_path = out_dir / "inventory.csv"
    pivot.to_csv(csv_path)
    print(f"\nInventory: {csv_path}")
    with pd.option_context("display.max_colwidth", 30, "display.width", 160):
        print(pivot.to_string())

    n_total = len(df_runs)
    n_finished = int((df_runs["state"] == "finished").sum())
    n_configs = df_runs["config_key"].nunique()
    n_seeds = df_runs["seed"].dropna().nunique()
    pct = 100.0 * n_finished / n_total if n_total else 0.0
    print(
        f"\nTotal runs: {n_total} | configs: {n_configs} | "
        f"distinct seeds: {n_seeds} | finished: {n_finished} ({pct:.1f}%)"
    )


def write_coverage_gaps(df_runs: pd.DataFrame, out_dir: Path) -> None:
    if df_runs.empty:
        return
    lines: list[str] = ["# Coverage gaps", ""]
    for dataset, sub in df_runs.groupby("dataset"):
        seeds_seen = sorted(int(s) for s in sub["seed"].dropna().unique().tolist())
        lines.append(f"## {dataset} (observed seeds: {seeds_seen})")
        for cfg_key, cfg_sub in sub.groupby("config_key"):
            states = {int(s): st for s, st in zip(cfg_sub["seed"], cfg_sub["state"], strict=True) if pd.notna(s)}
            present = set(states)
            missing = [s for s in seeds_seen if s not in present]
            non_finished = [(s, st) for s, st in states.items() if st != "finished"]
            n_finished = sum(1 for st in states.values() if st == "finished")
            parts = [f"finished={n_finished}"]
            if missing:
                parts.append(f"missing={missing}")
            if non_finished:
                parts.append(f"non_finished={non_finished}")
            lines.append(f"- `{cfg_key}` " + " ".join(parts))
        lines.append("")
    text = "\n".join(lines)
    path = out_dir / "coverage_gaps.md"
    path.write_text(text)
    print(f"Coverage: {path}")


def _label_short(config_key: str) -> str:
    """Drop dataset from `method|dataset|strategy[:notion]` to compact legend."""
    parts = config_key.split("|")
    if len(parts) == 3:
        method, _dataset, strat = parts
        return f"{method} / {strat}"
    return config_key


def plot_learning_curves(df_runs: pd.DataFrame, df_hist: pd.DataFrame, out_dir: Path) -> None:
    if df_hist.empty:
        print("(no history rows; skipping learning curves)")
        return
    merged = df_hist.merge(
        df_runs[["run_id", "config_key", "dataset", "seed"]],
        on="run_id",
        how="left",
    )
    for dataset, sub in merged.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(11, 6))
        cmap = plt.get_cmap("tab20")
        configs = sorted(sub["config_key"].dropna().unique())
        for i, cfg_key in enumerate(configs):
            csub = sub[sub["config_key"] == cfg_key]
            n_seeds = csub["seed"].nunique()
            grouped = csub.groupby("labeled_size")["test_accuracy"]
            mean = grouped.mean()
            std = grouped.std()
            xs = mean.index.to_numpy()
            color = cmap(i % 20)
            label = f"{_label_short(cfg_key)} (n={n_seeds})"
            ax.plot(xs, mean.to_numpy(), label=label, color=color, linewidth=1.4)
            if n_seeds > 1:
                ax.fill_between(
                    xs,
                    (mean - std).to_numpy(),
                    (mean + std).to_numpy(),
                    alpha=0.15,
                    color=color,
                )
        ax.set_xlabel("labeled samples")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"AL learning curves -- {dataset}")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"learning_curves_{dataset}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Curves: {path}")


def _strat_label(strategy: str | None, notion: str | None) -> str:
    if not strategy:
        return "?"
    return f"{strategy}:{notion}" if notion else strategy


_REFERENCE_STRATEGIES = ("margin", "random", "least_confident")
# (label, color, hatch) for the four bars per method, in plotting order.
_BAR_SERIES: tuple[tuple[str, str, str | None], ...] = (
    ("uncertainty:EU", "#1f77b4", None),
    ("uncertainty:TU", "#d62728", None),
    ("margin", "tab:green", "//"),
    ("random", "tab:gray", ".."),
)
_BASELINE_STYLES = {
    "margin": ("-", "tab:green"),
    "random": ("--", "tab:gray"),
    "least_confident": (":", "tab:orange"),
}


def _series_filter(sub: pd.DataFrame, label: str) -> pd.DataFrame:
    if label.startswith("uncertainty:"):
        notion = label.split(":", 1)[1]
        return sub[(sub["strategy"] == "uncertainty") & (sub["notion"] == notion)]
    return sub[sub["strategy"] == label]


def plot_nauc_bars(df_runs: pd.DataFrame, out_dir: Path) -> None:  # noqa: PLR0915
    """Per-method NAUC bars for uncertainty:EU/TU and margin/random, with `base` reference lines."""
    finished = df_runs[(df_runs["state"] == "finished") & df_runs["nauc"].notna()].copy()
    if finished.empty:
        print("(no finished runs with NAUC; skipping NAUC bars)")
        return

    for dataset, sub in finished.groupby("dataset"):
        baselines = {
            strat: sub[(sub["method_label"] == "base") & (sub["strategy"] == strat)]["nauc"].mean()
            for strat in _REFERENCE_STRATEGIES
        }

        # Methods on x-axis: anything with at least one uncertainty:* run.
        methods = sorted(sub[sub["strategy"] == "uncertainty"]["method_label"].dropna().unique())
        if not methods:
            print(f"(no uncertainty:* runs for {dataset}; skipping)")
            continue

        x = np.arange(len(methods))
        width = 0.8 / len(_BAR_SERIES)
        fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(methods)), 5.5))

        seed_counts: dict[str, dict[str, int]] = {m: {} for m in methods}
        # Track the lower edge of every drawn bar so we can compute a tight zoom y-min.
        bar_low_edges: list[float] = []

        for i, (label, color, hatch) in enumerate(_BAR_SERIES):
            series_df = _series_filter(sub, label)
            means: list[float] = []
            stds: list[float] = []
            for method in methods:
                pts = series_df[series_df["method_label"] == method]["nauc"].to_numpy()
                seed_counts[method][label] = len(pts)
                if len(pts):
                    m_val = float(np.mean(pts))
                    s_val = float(np.std(pts)) if len(pts) > 1 else 0.0
                    means.append(m_val)
                    stds.append(s_val)
                    bar_low_edges.append(min(m_val - s_val, float(np.min(pts))))
                else:
                    means.append(float("nan"))
                    stds.append(0.0)
            offset = (i - (len(_BAR_SERIES) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                label=label,
                color=color,
                hatch=hatch,
                capsize=2,
                edgecolor="black",
                linewidth=0.4,
            )
            for j, method in enumerate(methods):
                pts = series_df[series_df["method_label"] == method]["nauc"].to_numpy()
                if len(pts):
                    ax.scatter(
                        np.full(len(pts), x[j] + offset),
                        pts,
                        color="black",
                        s=8,
                        zorder=3,
                    )

        baseline_values: list[float] = []
        for strat, value in baselines.items():
            if pd.isna(value):
                continue
            ls, color = _BASELINE_STYLES[strat]
            ax.axhline(
                value, linestyle=ls, color=color, linewidth=1.2, alpha=0.7, label=f"base / {strat} ({value:.3f})"
            )
            baseline_values.append(float(value))

        def _xtick(method: str, counts_by_method: dict[str, dict[str, int]] = seed_counts) -> str:
            counts = counts_by_method[method]
            parts = [f"EU={counts.get('uncertainty:EU', 0)}"]
            if counts.get("uncertainty:TU", 0):
                parts.append(f"TU={counts['uncertainty:TU']}")
            parts.append(f"m={counts.get('margin', 0)}")
            parts.append(f"r={counts.get('random', 0)}")
            return f"{method}\nn: {' '.join(parts)}"

        ax.set_xticks(x)
        ax.set_xticklabels([_xtick(m) for m in methods], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("NAUC")
        ax.set_title(f"NAUC by method -- {dataset}  (uncertainty vs margin/random; `base` strategies as ref lines)")
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"nauc_{dataset}.png"
        fig.savefig(path, dpi=150)
        print(f"NAUC bars: {path}")

        # Zoomed variant: y starts a little below the lowest non-empty bar / point / baseline.
        if bar_low_edges:
            lo = min([*bar_low_edges, *baseline_values])
            top = ax.get_ylim()[1]
            pad = max(0.005, 0.05 * (top - lo))
            ax.set_ylim(lo - pad, top)
            ax.set_title(f"NAUC by method -- {dataset}  (zoomed; uncertainty vs margin/random)")
            zoom_path = out_dir / f"nauc_{dataset}_zoom.png"
            fig.savefig(zoom_path, dpi=150)
            print(f"NAUC bars (zoom): {zoom_path}")
        plt.close(fig)


def final_recap(df_runs: pd.DataFrame, out_dir: Path) -> None:
    print("\n=== Recap ===")
    if df_runs.empty:
        print("No runs.")
        return
    counts = df_runs["state"].value_counts().to_dict()
    print(f"State counts: {counts}")
    finished = df_runs[(df_runs["state"] == "finished") & df_runs["nauc"].notna()]
    if not finished.empty:
        top = (
            finished.groupby(["config_key", "dataset"])["nauc"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
            .head(5)
        )
        print("Top 5 (config, dataset) by mean NAUC:")
        print(top.to_string())
    print("\nArtifacts:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        metavar="ENTITY/PROJECT",
        help=(
            "wandb source to fetch from, in the form ENTITY/PROJECT. "
            "Pass multiple times to combine sources. Defaults: " + ", ".join(DEFAULT_SOURCES)
        ),
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore local cache and re-fetch from wandb.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    sources = args.source or list(DEFAULT_SOURCES)
    for s in sources:
        if s.count("/") != 1 or not all(s.split("/")):
            parser.error(f"--source must be ENTITY/PROJECT, got {s!r}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_runs, df_hist = load_or_fetch(sources, out_dir, args.refresh)
    print(f"\nLoaded {len(df_runs)} runs, {len(df_hist)} history rows.")
    if "source" in df_runs.columns and not df_runs.empty:
        per_source = df_runs.groupby("source").size().to_dict()
        print(f"Per-source counts: {per_source}")
    if df_runs.empty:
        return 0

    write_inventory(df_runs, out_dir)
    write_coverage_gaps(df_runs, out_dir)
    plot_learning_curves(df_runs, df_hist, out_dir)
    plot_nauc_bars(df_runs, out_dir)
    final_recap(df_runs, out_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
