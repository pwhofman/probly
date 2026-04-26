"""Tests for the prequential loop."""

from __future__ import annotations

import pytest

from river_uq.prequential import EXPECTED_COLUMNS, run_prequential


@pytest.mark.parametrize("method", ["arf", "deep_ensemble", "mc_dropout"])
def test_run_prequential_smoke(method: str) -> None:
    pytest.importorskip("torch")
    df = run_prequential(
        method=method, stream_name="stagger_drift", seed=0, n_steps=200
    )
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 200
    assert (df["epi"] >= 0).all()
    assert df["seed"].nunique() == 1
    assert df["method"].iloc[0] == method
    assert df["true_drift_t"].iloc[0] == 2000
    assert df["alarm_probly"].dtype == bool
    assert df["alarm_tailored"].dtype == bool


def test_stationary_stream_records_no_drift_t() -> None:
    df = run_prequential(
        method="arf", stream_name="agrawal_stationary", seed=0, n_steps=100
    )
    assert df["true_drift_t"].iloc[0] is None or pd_isna(df["true_drift_t"].iloc[0])


def pd_isna(v: object) -> bool:
    import pandas as pd

    return bool(pd.isna(v))
