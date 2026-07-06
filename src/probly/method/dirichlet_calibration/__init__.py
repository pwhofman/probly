"""Dirichlet calibration method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    DirichletCalibrationPredictor,
    dirichlet_calibration,
    dirichlet_calibration_generator,
)


@dirichlet_calibration_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "DirichletCalibrationPredictor",
    "dirichlet_calibration",
]
