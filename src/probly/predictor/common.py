"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from lazy_dispatch.singledispatch import lazydispatch


class Predictor[**In, Out](Protocol):
    """Protocol for generic predictors."""


class EnsemblePredictor[**In, Out](Predictor[In, Iterable[Out]], Iterable[Predictor[In, Out]], Protocol):
    """Protocol for ensemble predictors."""


class RandomPredictor[**In, Out](Predictor[In, Out], Protocol):
    """Protocol for non-deterministic predictors."""


@lazydispatch
def predict[**In, Out](predictor: Predictor[In, Out], *args: In.args, **kwargs: In.kwargs) -> Out:
    """Generic predict function."""
    if hasattr(predictor, "predict"):
        return predictor.predict(*args, **kwargs)  # type: ignore[no-any-return]
    if callable(predictor):
        return predictor(*args, **kwargs)
    msg = f"No predict function registered for type {type(predictor)}"
    raise NotImplementedError(msg)


@predict.register(list)
def predict_list[**In, Out](predictor: list[Predictor[In, Out]], *args: In.args, **kwargs: In.kwargs) -> list[Out]:
    """Predict for a list of predictors."""
    return [predict(p, *args, **kwargs) for p in predictor]
