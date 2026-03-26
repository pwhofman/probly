"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from lazy_dispatch import ProtocolRegistry, lazydispatch
from probly.representation.credal_set.common import CredalSet
from probly.representation.distribution.common import Distribution


class Predictor[**In, Out](ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for generic predictors."""


class RandomPredictor[**In, Out](Predictor[In, Out], Protocol):
    """Protocol for non-deterministic predictors."""


class IterablePredictor[**In, Out](Predictor[In, Iterable[Out]], Protocol):
    """Protocol for predictors that return an iterable of outputs."""


class DistributionPredictor[**In, Out: Distribution](Predictor[In, Out], Protocol):
    """Protocol for predictors that return a distribution over outputs."""


class CredalPredictor[**In, Out: CredalSet](Predictor[In, Out], Protocol):
    """Protocol for predictors that return a set of distributions over outputs."""


@lazydispatch
def predict[**In, Out](predictor: Predictor[In, Out], *args: In.args, **kwargs: In.kwargs) -> Out:
    """Generic predict function."""
    if hasattr(predictor, "predict"):
        return predictor.predict(*args, **kwargs)  # ty: ignore[call-non-callable]
    if callable(predictor):
        return predictor(*args, **kwargs)  # ty:ignore[call-top-callable, invalid-return-type]
    msg = f"No predict function registered for type {type(predictor)}"
    raise NotImplementedError(msg)
