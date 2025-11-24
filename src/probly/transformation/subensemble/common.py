"""Shared subensemble implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor


@lazydispatch
def subensemble_generator[In, KwIn, Out](base: Predictor[In, KwIn, Out]) -> Predictor[In, KwIn, Out]:
    """Generate a subensemble from a base model."""
    msg = f"No subensemble generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


def register(cls: LazyType, generator: Callable) -> None:
    """Register a class which can be used as a base for a subensemble."""
    subensemble_generator.register(cls=cls, func=generator)


def subensemble[T: Predictor](base: T, num_heads: int, reset_params: bool = True) -> T:
    """Create a subensemble predictor from a base predictor.

    Args:
        base: Predictor, The base model to be used for the subensemble.
        num_heads: int, The number of heads in the subensemble.
        reset_params: bool, Whether to reset the parameters for each head.

    Returns:
        Predictor, The subensemble predictor.
    """
    return subensemble_generator(base, num_heads=num_heads, reset_params=reset_params)
