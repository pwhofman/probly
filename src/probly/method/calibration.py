"""Calibration method compatibility exports."""

from __future__ import annotations

from probly.transformation.calibration import (
    CalibrationPredictor,
    dirichlet_calibration,
    isotonic_regression,
    platt_scaling,
    sklearn_identity_logit_estimator,
    temperature_scaling,
    torch_identity_logit_model,
    vector_scaling,
)

__all__ = [
    "CalibrationPredictor",
    "dirichlet_calibration",
    "isotonic_regression",
    "platt_scaling",
    "sklearn_identity_logit_estimator",
    "temperature_scaling",
    "torch_identity_logit_model",
    "vector_scaling",
]
