"""Shared ensemble implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_dispatch import lazy_singledispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor


@lazy_singledispatch
def ensemble_generator[In, KwIn, Out](base: Predictor[In, KwIn, Out]) -> Predictor[In, KwIn, Out]:
    """Generate an ensemble from a base model."""
    msg = f"No ensemble generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


def register(cls: LazyType, generator: Callable) -> None:
    """Register a class which can be used as a base for an ensemble."""
    ensemble_generator.register(cls=cls, func=generator)


def ensemble[T: Predictor](base: T, n_members: int) -> T:
    """Create an ensemble predictor from a base predictor.

    Args:
        base: Predictor, The base model to be used for the ensemble.
        n_members: The number of members in the ensemble.

    Returns:
        Predictor, The ensemble predictor.
    """
    return ensemble_generator(base, n_members=n_members)
