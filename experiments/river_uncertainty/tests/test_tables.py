"""Tests for latency table builder."""

from __future__ import annotations

import pandas as pd

from river_uq.tables import build_latency_table


def _mk_run(method: str, stream: str, seed: int, alarm_probly_t: int | None, alarm_tailored_t: int | None) -> pd.DataFrame:
    n = 100
    df = pd.DataFrame(
        {
            "t": range(n),
            "seed": seed,
            "method": method,
            "stream": stream,
            "y_true": 0,
            "y_pred": 0,
            "correct": 1,
            "total": 0.0,
            "alea": 0.0,
            "epi": 0.0,
            "alarm_probly": False,
            "alarm_tailored": False,
            "true_drift_t": 50,
        }
    )
    if alarm_probly_t is not None:
        df.loc[alarm_probly_t, "alarm_probly"] = True
    if alarm_tailored_t is not None:
        df.loc[alarm_tailored_t, "alarm_tailored"] = True
    return df


def test_latency_table_shape_and_columns() -> None:
    rows = pd.concat(
        [
            _mk_run("arf", "stagger_drift", seed=0, alarm_probly_t=55, alarm_tailored_t=60),
            _mk_run("arf", "stagger_drift", seed=1, alarm_probly_t=58, alarm_tailored_t=62),
        ],
        ignore_index=True,
    )
    table = build_latency_table(rows)
    assert set(table.columns) >= {
        "method", "stream", "probly_mean", "probly_std",
        "tailored_mean", "tailored_std", "tailored_label", "n_seeds", "n_missed_probly",
    }
    arf_row = table.query("method == 'arf' and stream == 'stagger_drift'").iloc[0]
    assert arf_row["probly_mean"] == 6.5  # ((55-50)+(58-50))/2
    assert arf_row["tailored_mean"] == 11.0


def test_missed_alarms_counted() -> None:
    rows = _mk_run("arf", "stagger_drift", seed=0, alarm_probly_t=None, alarm_tailored_t=60)
    table = build_latency_table(rows)
    arf_row = table.iloc[0]
    assert arf_row["n_missed_probly"] == 1
    assert pd.isna(arf_row["probly_mean"])
