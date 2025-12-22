"""Shared evidential classification implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor


@lazydispatch
def evidential_classification_appender[In, KwIn, Out](base: Predictor[In, KwIn, Out]) -> Predictor[In, KwIn, Out]:
    """Append an evidential classification activation function to a base model."""
    msg = f"No evidential classification appender registered for type {type(base)}"
    raise NotImplementedError(msg)


def register(cls: LazyType, appender: Callable) -> None:
    """Register a base model that the activation function will be appended to."""
    evidential_classification_appender.register(cls=cls, func=appender)


def evidential_classification[T: Predictor](base: T) -> T:
    """Create an evidential classification predictor from a base predictor based on :cite:`sensoyEvidentialDeep2018`.

    Args:
        base: Predictor, The base model to be used for evidential classification.

    Returns:
        Predictor, The evidential classification predictor.
    """
    return evidential_classification_appender(base)
