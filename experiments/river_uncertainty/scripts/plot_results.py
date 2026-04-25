"""Plot one stream's per-step trajectories from a run_experiment.py parquet.

The figure has one panel per method present in the parquet, with epistemic
uncertainty (median + IQR across seeds) on the left axis and rolling accuracy
on the right. A dashed vertical line marks ``true_drift_t`` if set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from river_uq.streams import STREAM_NAMES

SMOOTH_WIN: Final[int] = 50
EPI_COLOR: Final[str] = "#d62728"
ACC_COLOR: Final[str] = "#666666"
METHOD_LABEL: Final[dict[str, str]] = {
    "arf": "ARF",
    "deep_ensemble": "Deep Ensemble",
    "mc_dropout": "MC Dropout",
}


def _rolling(series: pd.Series, win: int = SMOOTH_WIN) -> pd.Series:
    return series.rolling(win, min_periods=1).mean()


def _aggregate(group: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        group.assign(rolled=lambda d: d.groupby("seed")[col].transform(_rolling))
        .groupby("t")["rolled"]
        .agg(med="median", lo=lambda s: s.quantile(0.25), hi=lambda s: s.quantile(0.75))
        .reset_index()
    )


def _draw_panel(ax: plt.Axes, group: pd.DataFrame, true_drift_t: int | None) -> None:
    epi = _aggregate(group, "epi")
    acc = _aggregate(group, "correct")

    ax.fill_between(
        epi["t"], epi["lo"], epi["hi"], color=EPI_COLOR, alpha=0.18, linewidth=0
    )
    ax.plot(epi["t"], epi["med"], color=EPI_COLOR, lw=1.4)
    ax.set_ylabel("epistemic", color=EPI_COLOR, fontsize=8)
    ax.tick_params(axis="y", labelcolor=EPI_COLOR, labelsize=7)
    ax.set_ylim(0, max(float(epi["hi"].max()), 1e-3) * 1.1)

    ax_right = ax.twinx()
    ax_right.plot(acc["t"], acc["med"], color=ACC_COLOR, lw=0.8, alpha=0.7)
    ax_right.set_ylim(0, 1.05)
    ax_right.set_ylabel("accuracy", color=ACC_COLOR, fontsize=8)
    ax_right.tick_params(axis="y", labelcolor=ACC_COLOR, labelsize=7)

    if true_drift_t is not None:
        ax.axvline(true_drift_t, color="black", linestyle="--", lw=0.8, alpha=0.5)

    ax.set_xlabel("step t", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)


def build_stream_figure(records: pd.DataFrame, stream: str) -> Figure:
    """Build a 1xK figure for one stream, one panel per method."""
    sub = records[records["stream"] == stream]
    if sub.empty:
        msg = f"no rows in parquet for stream={stream!r}"
        raise SystemExit(msg)

    methods = sorted(sub["method"].unique())
    fig, axes = plt.subplots(
        1, len(methods), figsize=(3.0 * len(methods), 2.6), sharey=False
    )
    axes_arr = np.atleast_1d(axes)

    true_t_raw = sub["true_drift_t"].iloc[0]
    true_t = None if pd.isna(true_t_raw) else int(true_t_raw)

    for ax, method in zip(axes_arr, methods, strict=True):
        panel = sub[sub["method"] == method]
        _draw_panel(ax, panel, true_t)
        ax.set_title(METHOD_LABEL.get(method, method), fontsize=9)

    fig.suptitle(stream, fontsize=10)
    fig.tight_layout()
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream",
        required=True,
        help=f"Stream to plot. One of: {', '.join(STREAM_NAMES)}.",
    )
    parser.add_argument(
        "--records-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "run_records.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write <stream>.{pdf,png}. Defaults to records-path's directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.stream not in STREAM_NAMES:
        msg = (
            f"unknown stream: {args.stream!r}. "
            f"Choose from: {', '.join(STREAM_NAMES)}."
        )
        raise SystemExit(msg)
    if not args.records_path.exists():
        msg = f"records parquet not found: {args.records_path}"
        raise SystemExit(msg)

    output_dir: Path = args.output_dir or args.records_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_parquet(args.records_path)
    fig = build_stream_figure(records, args.stream)

    pdf_path = output_dir / f"{args.stream}.pdf"
    png_path = output_dir / f"{args.stream}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    print(f"Wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
