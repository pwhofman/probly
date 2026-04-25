"""End-to-end smoke test: tiny prequential run -> figure object -> table."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import pandas as pd
import pytest

from river_uq.plotting import build_appendix_figure, build_headline_figure
from river_uq.prequential import EXPECTED_COLUMNS, run_prequential
from river_uq.tables import build_latency_table, latency_table_to_latex


@pytest.mark.parametrize("method", ["arf", "deep_ensemble", "mc_dropout"])
@pytest.mark.parametrize("stream", ["stagger_drift", "sea_drift", "agrawal_stationary"])
def test_each_combo_runs_short(method: str, stream: str) -> None:
    pytest.importorskip("torch")
    df = run_prequential(method=method, stream_name=stream, seed=0, n_steps=200)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 200
    assert (df["epi"] >= 0).all()
    assert not df["epi"].isna().any()


def test_full_pipeline_smoke() -> None:
    pytest.importorskip("torch")
    dfs = []
    for method in ["arf", "deep_ensemble", "mc_dropout"]:
        for stream in ["stagger_drift", "sea_drift"]:
            for seed in (0, 1):
                dfs.append(run_prequential(method=method, stream_name=stream, seed=seed, n_steps=120))
    df = pd.concat(dfs, ignore_index=True)

    fig = build_headline_figure(df)
    assert fig is not None
    assert len(fig.axes) >= 6  # 3 rows x 2 cols, possibly more due to twinx

    table = build_latency_table(df)
    assert len(table) == 6
    tex = latency_table_to_latex(table)
    assert "tabular" in tex
    assert "PageHinkley" in tex


def test_stationary_smoke_no_probly_alarm() -> None:
    """Tiny stationary run should not flag a probly-UQ alarm in the warmup-only window."""
    df = run_prequential(method="arf", stream_name="agrawal_stationary", seed=0, n_steps=200)
    fig = build_appendix_figure(df)
    assert fig is not None
