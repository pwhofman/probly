"""Tests for the prequential loop."""

from __future__ import annotations

from river_uq.prequential import EXPECTED_COLUMNS, run_prequential


def test_run_prequential_smoke() -> None:
    df = run_prequential(
        method="arf", stream_name="stagger_drift", seed=0, n_steps=200
    )
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 200
    assert (df["epi"] >= 0).all()
    assert df["seed"].nunique() == 1
    assert df["method"].iloc[0] == "arf"
    assert df["true_drift_t"].iloc[0] == 2000


def test_stationary_stream_records_no_drift_t() -> None:
    df = run_prequential(
        method="arf", stream_name="agrawal_stationary", seed=0, n_steps=100
    )
    assert df["true_drift_t"].iloc[0] is None or pd_isna(df["true_drift_t"].iloc[0])


def pd_isna(v: object) -> bool:
    import pandas as pd

    return bool(pd.isna(v))
