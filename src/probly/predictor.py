"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from typing import Protocol, TypeVar, Unpack, runtime_checkable

from lazy_dispatch.singledispatch import lazydispatch

In = TypeVar("In")
KwIn = TypeVar("KwIn")
Out = TypeVar("Out")


@runtime_checkable
class Predictor[In, KwIn, Out](Protocol):
    """Protocol for generic predictors."""

    def __call__(self, *args: In, **kwargs: Unpack[KwIn]) -> Out:
        """Call the wrapper with input data."""
        ...


@lazydispatch
def predict[In, KwIn, Out](predictor: Predictor[In, KwIn, Out], *args: In, **kwargs: Unpack[KwIn]) -> Out:
    """Generic predict function."""
    return predictor(*args, **kwargs)
