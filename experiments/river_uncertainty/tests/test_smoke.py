"""End-to-end smoke test: tiny prequential run -> tidy DataFrame."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import pandas as pd
import pytest

from river_uq.prequential import EXPECTED_COLUMNS, run_prequential


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
    """Concat across (method, stream, seed) produces a tidy frame the plotter can read."""
    pytest.importorskip("torch")
    dfs = []
    for method in ["arf", "deep_ensemble", "mc_dropout"]:
        for stream in ["stagger_drift", "sea_drift"]:
            for seed in (0, 1):
                dfs.append(
                    run_prequential(method=method, stream_name=stream, seed=seed, n_steps=120)
                )
    df = pd.concat(dfs, ignore_index=True)

    assert list(df.columns) == EXPECTED_COLUMNS
    assert df["method"].nunique() == 3
    assert df["stream"].nunique() == 2
    assert df["seed"].nunique() == 2
    assert len(df) == 3 * 2 * 2 * 120
