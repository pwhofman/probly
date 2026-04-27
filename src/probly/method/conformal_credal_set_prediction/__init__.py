"""Conformalized Credal Set Prediction implementation."""

from __future__ import annotations
from probly.lazy_types import TORCH_MODULE

from probly.method.conformal_credal_set_prediction._common import (
    ConformalCredalSetPredictor,
    conformal_credal_set_generator,
    conformal_total_variation,
)

@conformal_credal_set_generator.delayed_register(TORCH_MODULE)
def _(_cls: type[object]) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "ConformalCredalSetPredictor",
    "conformal_total_variation",
    "conformal_credal_set_generator"
]
