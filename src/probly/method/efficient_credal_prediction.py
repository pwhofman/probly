"""Efficient credal prediction method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.representation.distribution import CategoricalDistribution
from probly.transformation.credal_bounds import CredalBoundsPredictor, credal_bounds


@runtime_checkable
class EfficientCredalPredictor[**In, Out: CategoricalDistribution](CredalBoundsPredictor[In, Out], Protocol):
    """A predictor routed through the efficient credal prediction representer."""


efficient_credal_prediction = EfficientCredalPredictor.register_factory(credal_bounds)

__all__ = ["EfficientCredalPredictor", "efficient_credal_prediction"]
