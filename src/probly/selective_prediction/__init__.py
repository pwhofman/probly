"""Selective prediction: predict or abstain based on the uncertainty of an uncertainty-aware model."""

from __future__ import annotations

from ._common import (
    SelectivePrediction,
    SelectivePredictor,
    ThresholdSelectivePredictor,
)

__all__ = [
    "SelectivePrediction",
    "SelectivePredictor",
    "ThresholdSelectivePredictor",
]
