"""Selective prediction: predict or abstain based on the uncertainty of an uncertainty-aware model."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR

from ._common import (
    CoverageSelector,
    SelectivePrediction,
    SelectivePredictor,
    Selector,
    SGRSelector,
    ThresholdSelector,
    _to_float64_numpy,
)


@_to_float64_numpy.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "CoverageSelector",
    "SGRSelector",
    "SelectivePrediction",
    "SelectivePredictor",
    "Selector",
    "ThresholdSelector",
]
