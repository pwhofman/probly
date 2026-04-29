"""Benchmark-local base-model marker method."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.method.cast import cast
from probly.predictor import Predictor


@runtime_checkable
class BasePredictor[**In, Out](Predictor[In, Out], Protocol, structural_checking=False):
    """Benchmark marker for an unmodified base predictor."""


base = BasePredictor.register_factory(cast)

__all__ = ["BasePredictor", "base"]
