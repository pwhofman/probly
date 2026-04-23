"""Calibration transformations for logit predictors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.lazy_types import SKLEARN_MODULE, TORCH_MODULE

from ._common import (
    CalibrationPredictor,
    calibration_generator,
    isotonic_regression,
    platt_scaling,
    temperature_scaling,
    vector_scaling,
)

if TYPE_CHECKING:
    from .sklearn import SklearnIdentityLogitEstimator
    from .torch import TorchIdentityLogitModel


@calibration_generator.delayed_register(TORCH_MODULE)
def _(_: type[object]) -> None:
    from . import torch as torch  # noqa: PLC0415


@calibration_generator.delayed_register(SKLEARN_MODULE)
def _(_: type[object]) -> None:
    from . import sklearn as sklearn  # noqa: PLC0415


def torch_identity_logit_model() -> TorchIdentityLogitModel:
    """Create a torch pass-through model for direct logit calibration."""
    from .torch import TorchIdentityLogitModel  # noqa: PLC0415

    return TorchIdentityLogitModel()


def sklearn_identity_logit_estimator() -> SklearnIdentityLogitEstimator:
    """Create a sklearn pass-through estimator for direct logit calibration."""
    from .sklearn import SklearnIdentityLogitEstimator  # noqa: PLC0415

    return SklearnIdentityLogitEstimator()


__all__ = [
    "CalibrationPredictor",
    "isotonic_regression",
    "platt_scaling",
    "sklearn_identity_logit_estimator",
    "temperature_scaling",
    "torch_identity_logit_model",
    "vector_scaling",
]
