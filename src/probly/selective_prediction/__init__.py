"""Selective prediction: predict or abstain based on the uncertainty of an uncertainty-aware model.

A :func:`selective_predictor` wraps the :class:`~probly.representer.Representer` of a model that probly can represent,
e.g. an ensemble, an MC-dropout model, or a plain classifier declared with :func:`~probly.method.cast`, and accepts or
rejects each prediction with a :class:`~probly.selective_prediction.selectors.Selector`, a rule on the uncertainty
criterion.
"""

from __future__ import annotations

from ._common import SelectivePrediction, SelectivePredictor, selective_predictor
from .selectors import Selector, selector

__all__ = [
    "SelectivePrediction",
    "SelectivePredictor",
    "Selector",
    "selective_predictor",
    "selector",
]
