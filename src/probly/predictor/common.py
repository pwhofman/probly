"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, Unpack, runtime_checkable

from lazy_dispatch.singledispatch import lazydispatch


@runtime_checkable
class Predictor[In, KwIn, Out](Protocol):
    """Protocol for generic predictors."""


@runtime_checkable
class EnsemblePredictor[In, KwIn, Out](
    Predictor[In, KwIn, Iterable[Out]], Iterable[Predictor[In, KwIn, Out]], Protocol
):
    """Protocol for ensemble predictors."""


@lazydispatch
def predict[In, KwIn, Out](predictor: Predictor[In, KwIn, Out], *args: In, **kwargs: Unpack[KwIn]) -> Out:
    """Generic predict function."""
    if hasattr(predictor, "predict"):
        return predictor.predict(*args, **kwargs)  # type: ignore[no-any-return]
    if callable(predictor):
        return predictor(*args, **kwargs)  # type: ignore[no-any-return]
    msg = f"No predict function registered for type {type(predictor)}"
    raise NotImplementedError(msg)


@predict.register(list)
def predict_list(predictor: list, *args: Any, **kwargs: Any) -> list:  # noqa: ANN401
    """Predict for a list of predictors."""
    return [predict(p, *args, **kwargs) for p in predictor]
