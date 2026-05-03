"""Conformal transformations for regression and classification."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, SKLEARN_MODULE, TORCH_MODULE

from ._common import (
    AbsoluteErrorConformalSetPredictor,
    APSConformalSetPredictor,
    ClassificationConformalSetPredictor,
    ConformalSetPredictor,
    CQRConformalSetPredictor,
    CQRrConformalSetPredictor,
    LACConformalSetPredictor,
    RAPSConformalSetPredictor,
    RegressionConformalSetPredictor,
    SAPSConformalSetPredictor,
    UACQRConformalSetPredictor,
    conformal_absolute_error,
    conformal_aps,
    conformal_cqr,
    conformal_cqr_r,
    conformal_generator,
    conformal_lac,
    conformal_raps,
    conformal_saps,
    conformal_uacqr,
)


@conformal_generator.delayed_register(TORCH_MODULE)
def _(_cls: type[object]) -> None:
    from . import torch as torch  # noqa: PLC0415


@conformal_generator.delayed_register(FLAX_MODULE)
def _(_cls: type[object]) -> None:
    from . import flax as flax  # noqa: PLC0415


@conformal_generator.delayed_register(SKLEARN_MODULE)
def _(_cls: type[object]) -> None:
    from . import sklearn as sklearn  # noqa: PLC0415


__all__ = [
    "APSConformalSetPredictor",
    "AbsoluteErrorConformalSetPredictor",
    "CQRConformalSetPredictor",
    "CQRrConformalSetPredictor",
    "ClassificationConformalSetPredictor",
    "ConformalSetPredictor",
    "LACConformalSetPredictor",
    "RAPSConformalSetPredictor",
    "RegressionConformalSetPredictor",
    "SAPSConformalSetPredictor",
    "UACQRConformalSetPredictor",
    "conformal_absolute_error",
    "conformal_aps",
    "conformal_cqr",
    "conformal_cqr_r",
    "conformal_lac",
    "conformal_raps",
    "conformal_saps",
    "conformal_uacqr",
]
