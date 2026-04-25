"""Tests for latency table builder."""

from __future__ import annotations

import pandas as pd

from river_uq.tables import build_latency_table, latency_table_to_latex


def _mk_run(
    method: str,
    stream: str,
    seed: int,
    alarm_probly_t: int | None,
    alarm_tailored_t: int | None,
    true_drift_t: int = 50,
) -> pd.DataFrame:
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
            "true_drift_t": true_drift_t,
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
        "method", "stream", "n_seeds",
        "probly_tp_mean", "probly_tp_std", "probly_n_tp", "probly_n_fa", "probly_n_miss",
        "tailored_tp_mean", "tailored_tp_std", "tailored_n_tp", "tailored_n_fa", "tailored_n_miss",
        "tailored_label",
    }
    arf_row = table.query("method == 'arf' and stream == 'stagger_drift'").iloc[0]
    assert arf_row["probly_tp_mean"] == 6.5  # ((55-50)+(58-50))/2
    assert arf_row["probly_n_tp"] == 2
    assert arf_row["probly_n_fa"] == 0
    assert arf_row["tailored_tp_mean"] == 11.0


def test_missed_alarms_counted() -> None:
    rows = _mk_run("arf", "stagger_drift", seed=0, alarm_probly_t=None, alarm_tailored_t=60)
    table = build_latency_table(rows)
    arf_row = table.iloc[0]
    assert arf_row["probly_n_miss"] == 1
    assert arf_row["probly_n_tp"] == 0
    assert pd.isna(arf_row["probly_tp_mean"])


def test_false_alarm_split() -> None:
    """Alarm before true_drift_t counts as FA, not as negative-latency TP."""
    rows = pd.concat(
        [
            _mk_run("arf", "sea_drift", seed=0, alarm_probly_t=60, alarm_tailored_t=40),
            _mk_run("arf", "sea_drift", seed=1, alarm_probly_t=55, alarm_tailored_t=30),
            _mk_run("arf", "sea_drift", seed=2, alarm_probly_t=None, alarm_tailored_t=70),
        ],
        ignore_index=True,
    )
    table = build_latency_table(rows)
    r = table.iloc[0]
    # probly: 2 TPs at t=60,55 -> latencies 10, 5; 0 FA; 1 miss
    assert r["probly_n_tp"] == 2
    assert r["probly_n_fa"] == 0
    assert r["probly_n_miss"] == 1
    assert r["probly_tp_mean"] == 7.5
    # tailored: 2 FAs (t=40, 30 < 50), 1 TP at t=70
    assert r["tailored_n_tp"] == 1
    assert r["tailored_n_fa"] == 2
    assert r["tailored_n_miss"] == 0
    assert r["tailored_tp_mean"] == 20.0


def test_latex_renders_tp_and_fa_counts() -> None:
    rows = pd.concat(
        [
            _mk_run("arf", "sea_drift", seed=0, alarm_probly_t=60, alarm_tailored_t=40),
            _mk_run("arf", "sea_drift", seed=1, alarm_probly_t=None, alarm_tailored_t=30),
        ],
        ignore_index=True,
    )
    tex = latency_table_to_latex(build_latency_table(rows))
    assert "tabular" in tex
    assert "TP" in tex
    assert "FA" in tex
