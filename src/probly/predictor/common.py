"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from lazy_dispatch import ProtocolRegistry, lazydispatch
from probly.representation.credal_set.common import CredalSet
from probly.representation.distribution.common import Distribution

### Predictor Protocols


class Predictor[**In, Out](ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for generic predictors."""


class EnsemblePredictor[**In, Out](Predictor[In, Iterable[Out]], Iterable[Predictor[In, Out]], Protocol):
    """Protocol for ensemble predictors."""


class RandomPredictor[**In, Out](Predictor[In, Out], Protocol):
    """Protocol for non-deterministic predictors."""


class DistributionPredictor[**In, Out: Distribution](Predictor[In, Out], Protocol):
    """Protocol for predictors that return a distribution over outputs."""


class CredalPredictor[**In, Out: CredalSet](Predictor[In, Out], Protocol):
    """Protocol for predictors that return a set of distributions over outputs."""


### Predictor protocol registrations

EnsemblePredictor.register(list)

### Generic predict function


@lazydispatch
def predict[**In, Out](predictor: Predictor[In, Out], *args: In.args, **kwargs: In.kwargs) -> Out:
    """Generic predict function."""
    if hasattr(predictor, "predict"):
        return predictor.predict(*args, **kwargs)  # ty: ignore[call-non-callable]
    if callable(predictor):
        return predictor(*args, **kwargs)  # ty:ignore[call-top-callable, invalid-return-type]
    msg = f"No predict function registered for type {type(predictor)}"
    raise NotImplementedError(msg)


@predict.register(EnsemblePredictor)
def predict_list[**In, Out](predictor: EnsemblePredictor[In, Out], *args: In.args, **kwargs: In.kwargs) -> list[Out]:
    """Predict for a list of predictors."""
    return [predict(p, *args, **kwargs) for p in predictor]
