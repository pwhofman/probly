"""Credal net method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.representation.credal_set import ProbabilityIntervalsCredalSet
from probly.transformation.interval_classifier import IntervalClassifierPredictor, interval_classifier


@runtime_checkable
class CredalNetPredictor[**In, Out: ProbabilityIntervalsCredalSet](IntervalClassifierPredictor[In, Out], Protocol):
    """A predictor routed through the credal net method API."""


credal_net = CredalNetPredictor.register_factory(interval_classifier)
credal_net.__doc__ = """Create a Credal Net predictor from a base predictor based on :cite:`saleSecondOrder2024`.

Args:
    base: Predictor, The base classifier whose layers are replaced with interval counterparts.
    use_base_weights: bool, If True, copy each replaced layer's weights into the center slots. Default is False.

Returns:
    CredalNetPredictor, The credal net predictor outputting a ProbabilityIntervalsCredalSet.
"""

__all__ = ["CredalNetPredictor", "credal_net"]
