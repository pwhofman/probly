"""Credal net method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.representation.credal_set import ProbabilityIntervalsCredalSet
from probly.transformation.interval_classifier import IntervalClassifierPredictor, interval_classifier


@runtime_checkable
class CredalNetPredictor[**In, Out: ProbabilityIntervalsCredalSet](IntervalClassifierPredictor[In, Out], Protocol):
    """A predictor routed through the credal net method API."""


credal_net = CredalNetPredictor.register_factory(interval_classifier)

__all__ = ["CredalNetPredictor", "credal_net"]
