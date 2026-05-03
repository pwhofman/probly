"""Prequential loop: one (method, stream, seed) run -> tidy DataFrame."""

from __future__ import annotations

from typing import Any, Final, cast

import pandas as pd

from river_uq.models import build_model
from river_uq.streams import build_stream

EXPECTED_COLUMNS: Final[list[str]] = [
    "t",
    "seed",
    "method",
    "stream",
    "y_true",
    "y_pred",
    "correct",
    "total",
    "alea",
    "epi",
    "true_drift_t",
]


def run_prequential(
    *,
    method: str,
    stream_name: str,
    seed: int,
    n_steps: int = 3000,
) -> pd.DataFrame:
    """Run one prequential loop and return per-step records.

    Args:
        method: Model kind forwarded to :func:`build_model`. Paper uses ``"arf"``.
        stream_name: Stream name forwarded to :func:`build_stream`.
        seed: Seed for both stream and model.
        n_steps: Number of stream samples to consume.

    Returns:
        DataFrame with columns matching :data:`EXPECTED_COLUMNS`.
    """
    model = build_model(method, seed=seed)
    stream, true_drift_t = build_stream(stream_name, seed=seed, n=n_steps)

    rows: list[dict] = []
    for t, (x, y) in enumerate(stream):
        # test-then-train: predict on x BEFORE updating
        y_pred = model.predict_one(x)
        decomp = model.epistemic_decomposition(x)
        correct = int(int(y_pred) == int(y))
        model.learn_one(x, y)

        rows.append(
            {
                "t": t,
                "seed": seed,
                "method": method,
                "stream": stream_name,
                "y_true": int(y),
                "y_pred": int(y_pred),
                "correct": correct,
                "total": float(decomp.total),
                "alea": float(decomp.aleatoric),
                "epi": float(decomp.epistemic),
                "true_drift_t": true_drift_t,
            }
        )

    return pd.DataFrame(rows, columns=cast("Any", EXPECTED_COLUMNS))
