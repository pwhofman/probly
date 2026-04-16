"""Conformal transformations for regression and classification."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    ConformalClassificationCalibrator,
    ConformalQuantileRegressionCalibrator,
    ConformalRegressionCalibrator,
    conformalize_classifier,
    conformalize_quantile_regressor,
    conformalize_regressor,
    ensure_distribution_2d,
)


@ensure_distribution_2d.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_cls: type[object]) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "ConformalClassificationCalibrator",
    "ConformalQuantileRegressionCalibrator",
    "ConformalRegressionCalibrator",
    "conformalize_classifier",
    "conformalize_quantile_regressor",
    "conformalize_regressor",
]
