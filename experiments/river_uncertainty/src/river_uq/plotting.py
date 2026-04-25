"""Headline and appendix figure builders.

The figures are computed from the tidy per-step DataFrame produced by
``run_prequential``. Aggregation across seeds: median + IQR (25/75th
percentiles) for the time series; median alarm time across seeds for the
detector markers.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

METHOD_ORDER = ["arf", "deep_ensemble", "mc_dropout"]
METHOD_LABEL = {
    "arf": "ARF",
    "deep_ensemble": "Deep Ensemble",
    "mc_dropout": "MC Dropout",
}
STREAM_LABEL = {
    "stagger_drift": "STAGGER (abrupt, clean)",
    "sea_drift": "SEA (abrupt, noisy)",
    "agrawal_stationary": "Agrawal (stationary)",
}
SMOOTH_WIN = 50
PROBLY_COLOR = "#d62728"  # red
TAILORED_COLOR = "#1f77b4"  # blue


def _rolling(series: pd.Series, win: int = SMOOTH_WIN) -> pd.Series:
    return series.rolling(win, min_periods=1).mean()


def _aggregate(group: pd.DataFrame, col: str) -> pd.DataFrame:
    """Aggregate `col` across seeds: median + IQR per t."""
    rolled = (
        group.assign(rolled=lambda d: d.groupby("seed")[col].transform(_rolling))
        .groupby("t")["rolled"]
        .agg(med="median", lo=lambda s: s.quantile(0.25), hi=lambda s: s.quantile(0.75))
        .reset_index()
    )
    return rolled


def _alarm_times(group: pd.DataFrame, col: str) -> np.ndarray:
    times: list[int] = []
    for _, sg in group.groupby("seed"):
        fired = sg.loc[sg[col], "t"]
        if len(fired):
            times.append(int(fired.iloc[0]))
    return np.asarray(times, dtype=float)


def _draw_cell(ax_left, group: pd.DataFrame, true_drift_t: int | None) -> None:
    epi = _aggregate(group, "epi")
    acc = _aggregate(group, "correct")

    ax_left.fill_between(epi["t"], epi["lo"], epi["hi"], color=PROBLY_COLOR, alpha=0.18, linewidth=0)
    ax_left.plot(epi["t"], epi["med"], color=PROBLY_COLOR, lw=1.4, label="epistemic (median)")
    ax_left.set_ylabel("epistemic", color=PROBLY_COLOR, fontsize=8)
    ax_left.tick_params(axis="y", labelcolor=PROBLY_COLOR, labelsize=7)

    ax_right = ax_left.twinx()
    ax_right.plot(acc["t"], acc["med"], color="#666666", lw=0.8, alpha=0.7)
    ax_right.set_ylim(0, 1.05)
    ax_right.set_ylabel("accuracy", color="#666666", fontsize=8)
    ax_right.tick_params(axis="y", labelcolor="#666666", labelsize=7)

    if true_drift_t is not None:
        ax_left.axvline(true_drift_t, color="black", linestyle="--", lw=0.8, alpha=0.5)

    probly_t = _alarm_times(group, "alarm_probly")
    tailored_t = _alarm_times(group, "alarm_tailored")
    ymax = max(epi["hi"].max(), 1e-3)
    if len(probly_t):
        med = float(np.median(probly_t))
        q25, q75 = float(np.quantile(probly_t, 0.25)), float(np.quantile(probly_t, 0.75))
        ax_left.errorbar(
            [med], [ymax * 1.05],
            xerr=[[med - q25], [q75 - med]],
            fmt="v", color=PROBLY_COLOR, markersize=7, capsize=2, lw=1, label="probly-UQ alarm",
        )
    if len(tailored_t):
        med = float(np.median(tailored_t))
        q25, q75 = float(np.quantile(tailored_t, 0.25)), float(np.quantile(tailored_t, 0.75))
        ax_left.errorbar(
            [med], [ymax * 1.15],
            xerr=[[med - q25], [q75 - med]],
            fmt="^", color=TAILORED_COLOR, markersize=7, capsize=2, lw=1, label="tailored alarm",
        )
    ax_left.set_ylim(0, ymax * 1.25)
    ax_left.set_xlabel("step t", fontsize=8)
    ax_left.tick_params(axis="x", labelsize=7)


def build_headline_figure(records: pd.DataFrame) -> Figure:
    """Build the 3x2 headline figure (methods x drift streams)."""
    streams = ["stagger_drift", "sea_drift"]
    methods = [m for m in METHOD_ORDER if m in records["method"].unique()]
    fig, axes = plt.subplots(
        len(methods), len(streams),
        figsize=(8.5, 2.4 * len(methods)),
        sharex=True,
    )
    if len(methods) == 1:
        axes = np.array([axes])
    for r, method in enumerate(methods):
        for c, stream in enumerate(streams):
            ax = axes[r, c]
            sub = records.query("method == @method and stream == @stream")
            if len(sub) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue
            true_t = sub["true_drift_t"].iloc[0]
            true_t_val = None if pd.isna(true_t) else int(true_t)
            _draw_cell(ax, sub, true_t_val)
            if r == 0:
                ax.set_title(STREAM_LABEL.get(stream, stream), fontsize=9)
            if c == 0:
                ax.text(
                    -0.18, 0.5, METHOD_LABEL.get(method, method),
                    transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10, fontweight="bold",
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    return fig


def build_appendix_figure(records: pd.DataFrame) -> Figure:
    """Build the 1x3 stationary-stream appendix figure."""
    methods = [m for m in METHOD_ORDER if m in records["method"].unique()]
    fig, axes = plt.subplots(1, len(methods), figsize=(8.5, 2.4), sharex=True, sharey=True)
    if len(methods) == 1:
        axes = np.array([axes])
    for c, method in enumerate(methods):
        ax = axes[c]
        sub = records.query("method == @method and stream == 'agrawal_stationary'")
        if len(sub) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        _draw_cell(ax, sub, None)
        ax.set_title(METHOD_LABEL.get(method, method), fontsize=9)
    fig.suptitle("Agrawal stationary -- no probly-UQ alarms expected", fontsize=10)
    fig.tight_layout()
    return fig
