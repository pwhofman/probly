"""Shared implementation of Prior Networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.transformation.evidential import evidential_classification

if TYPE_CHECKING:
    from probly.predictor import Predictor


def prior_network[**In, Out](base: Predictor[In, Out]) -> Predictor[In, Out]:
    """Create a Prior Network predictor from base model based on :cite:`malininPredictiveUncertaintyEstimation2018`."""
    return evidential_classification(base)
