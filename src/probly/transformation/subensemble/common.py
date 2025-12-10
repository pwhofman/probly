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


def subensemble[T: Predictor](
    base: T,
    num_heads: int,
    head: T | None = None,
    reset_params: bool = True,
    head_layer: int = 1,
) -> T:
    """Create a subensemble predictor from a base model or a base model and head model.

    Supported configurations:
        1. base only:
            Head is created by extracting the last `head_layer` layers of `base`,
            while the remaining layers are used to create the backbone.

        2. base and head:
            `base` is used as the shared backbone,
            `head` is duplicated `num_heads` times to form the subensemble heads.

    Args:
        base: Predictor, The model to be used as backbone or to create the backbone and heads.
        num_heads: int, The number of heads in the subensemble.
        head: Predictor, Optional model to be used as head of the subensemble.
        reset_params: bool, Whether to reset the parameters of each head.
        head_layer: int, Optional the number of layers used to create the head if no head model is provided.

    Returns:
        Predictor, The subensemble predictor.

    Raises:
        ValueError: If `head_layer` or `num_heads` is not a positive integer.
    """
    if head_layer <= 0:
        msg = f"head_layer must be a positive number, but got head_layer={head_layer} instead."
        raise ValueError(msg)
    if num_heads <= 0:
        msg = f"num_heads must be positive number, but got num_heads={num_heads} instead."
        raise ValueError(msg)
    return subensemble_generator(base, num_heads=num_heads, head=head, reset_params=reset_params, head_layer=head_layer)
