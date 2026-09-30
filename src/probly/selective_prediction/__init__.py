"""Selective prediction: predict or abstain based on the uncertainty of an uncertainty-aware model."""

from __future__ import annotations

from ._common import (
    SelectivePrediction,
    SelectivePredictor,
    Selector,
    ThresholdSelector,
)

__all__ = [
    "SelectivePrediction",
    "SelectivePredictor",
    "Selector",
    "ThresholdSelector",
]
