"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from typing import Protocol, Unpack, runtime_checkable

from lazy_dispatch.singledispatch import lazy_singledispatch


@runtime_checkable
class Predictor[In, KwIn, Out](Protocol):
    """Protocol for generic predictors."""

    def __call__(self, *args: In, **kwargs: Unpack[KwIn]) -> Out:
        """Call the wrapper with input data."""
        ...


@lazy_singledispatch
def predict[In, KwIn, Out](predictor: Predictor[In, KwIn, Out], *args: In, **kwargs: Unpack[KwIn]) -> Out:
    """Generic predict function."""
    return predictor(*args, **kwargs)
