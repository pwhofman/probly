"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

from lazy_dispatch import ProtocolRegistry, lazydispatch
from probly.representation.credal_set._common import CredalSet
from probly.representation.distribution._common import Distribution


@runtime_checkable
class Predictor[**In, Out](ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for generic predictors."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        predict_method = getattr(subclass, "predict", None)
        if predict_method is not None and callable(predict_method):
            return True
        if issubclass(subclass, Callable):
            return True
        return NotImplemented


@runtime_checkable
class RandomPredictor[**In, Out](Predictor[In, Out], Protocol):
    """Protocol for non-deterministic predictors."""


@runtime_checkable
class IterablePredictor[**In, Out](Predictor[In, Iterable[Out]], Protocol):
    """Protocol for predictors that return an iterable of outputs."""


@runtime_checkable
class DistributionPredictor[**In, Out: Distribution](Predictor[In, Out], Protocol):
    """Protocol for predictors that return a distribution over outputs."""


@runtime_checkable
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
