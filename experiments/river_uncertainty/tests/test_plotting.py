"""Tests for plotting (smoke only)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import pandas as pd
import pytest

from river_uq.plotting import build_appendix_figure, build_headline_figure
from river_uq.prequential import EXPECTED_COLUMNS


def _make_df(methods: list[str], streams: list[str], n: int = 100) -> pd.DataFrame:
    rows: list[dict] = []
    for m in methods:
        for s in streams:
            for seed in range(2):
                for t in range(n):
                    rows.append(
                        {
                            "t": t,
                            "seed": seed,
                            "method": m,
                            "stream": s,
                            "y_true": 0,
                            "y_pred": 0,
                            "correct": 1,
                            "total": 0.1,
                            "alea": 0.05,
                            "epi": 0.05 + (0.5 if t > n // 2 else 0.0),
                            "alarm_probly": (t == n // 2 + 5),
                            "alarm_tailored": (t == n // 2 + 10),
                            "true_drift_t": n // 2,
                        }
                    )
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_headline_figure_returns_3x2_figure() -> None:
    df = _make_df(
        ["arf", "deep_ensemble", "mc_dropout"],
        ["stagger_drift", "sea_drift"],
    )
    fig = build_headline_figure(df)
    assert fig is not None
    axes = fig.axes
    assert len(axes) >= 6  # 3 rows x 2 cols (left axes only -- right axes are twins)


def test_appendix_figure_runs() -> None:
    df = _make_df(["arf", "deep_ensemble", "mc_dropout"], ["agrawal_stationary"])
    fig = build_appendix_figure(df)
    assert fig is not None
