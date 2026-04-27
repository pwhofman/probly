"""Prequential loop: one (method, stream, seed) run -> tidy DataFrame."""

from __future__ import annotations

from typing import Any, Final, cast

import pandas as pd

from river_uq.detectors import (
    ARFNativeDetector,
    Detector,
    PageHinkleyErrorDetector,
    ProblyUQDetector,
)
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
    "alarm_probly",
    "alarm_tailored",
    "true_drift_t",
]


def _make_tailored(method: str) -> Detector:
    if method == "arf":
        return ARFNativeDetector(warmup_end=1500)
    return PageHinkleyErrorDetector(min_instances=500, threshold=50.0, warmup_end=1500)


def run_prequential(
    *,
    method: str,
    stream_name: str,
    seed: int,
    n_steps: int = 3000,
) -> pd.DataFrame:
    """Run one prequential loop and return per-step records.

    Args:
        method: ``"arf"``, ``"deep_ensemble"``, or ``"mc_dropout"``.
        stream_name: Stream name forwarded to :func:`build_stream`.
        seed: Seed for both stream and model.
        n_steps: Number of stream samples to consume.

    Returns:
        DataFrame with columns matching :data:`EXPECTED_COLUMNS`.
    """
    model = build_model(method, seed=seed)
    stream, true_drift_t = build_stream(stream_name, seed=seed, n=n_steps)
    probly_det = ProblyUQDetector()
    tailored_det = _make_tailored(method)

    rows: list[dict] = []
    for t, (x, y) in enumerate(stream):
        # test-then-train: predict on x BEFORE updating
        y_pred = model.predict_one(x)
        decomp = model.epistemic_decomposition(x)
        correct = int(int(y_pred) == int(y))
        error = 1 - correct
        n_drifts = int(getattr(model, "n_drifts_detected", 0))

        alarm_probly = probly_det.update(
            t, epi=decomp.epistemic, error=error, model_n_drifts=n_drifts
        )
        model.learn_one(x, y)
        post_n_drifts = int(getattr(model, "n_drifts_detected", 0))
        alarm_tailored = tailored_det.update(
            t, epi=decomp.epistemic, error=error, model_n_drifts=post_n_drifts
        )

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
                "alarm_probly": bool(alarm_probly),
                "alarm_tailored": bool(alarm_tailored),
                "true_drift_t": true_drift_t,
            }
        )

    return pd.DataFrame(rows, columns=cast("Any", EXPECTED_COLUMNS))
