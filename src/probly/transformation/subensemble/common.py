"""Shared subensemble implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from lazy_dispatch import lazydispatch
from probly.transformation.ensemble import ensemble

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
    num_heads: int,
    base: T | None = None,
    head: T | None = None,
    reset_params: bool = True,
    head_layer: int = 1,
) -> T:
    """Create a subensemble predictor from a base model, a head model or both.

    Exactly one of `base` or `head` must be provided, unless both are provided.

    Supported configurations:
        1. base only:
            Head is created by extracting the last `head_layer` layers of `base`,
            while the remaining layers are used to create the backbone.

        2. head only:
            Subensemble consists of `num_heads` independent copies of `head`.

        3. base and head:
            `base` is used as the shared backbone,
            `head` is duplicated `num_heads` times to form the subensemble heads.

    Args:
        num_heads: int, The number of heads in the subensemble.
        base: Predictor, Optional model to be used as described in configurations.
        head: Predictor, Optional model to be used as head of the subensemble.
        reset_params: bool, Whether to reset the parameters of each head.
        head_layer: int, Optional the number of layers used to create the head if no head model is provided.

    Returns:
        Predictor, The subensemble predictor.

    Raises:
        ValueError: If neither `base` nor `head` are provided.
        NotImplementedError: If `head` is not of instance(nn.Module).
    """
    if base is None:
        if head is None:
            msg = "Either base, head or both must be provided."
            raise ValueError(msg)
        if isinstance(head, nn.Module):
            return ensemble(head, num_members=num_heads, reset_params=reset_params)  # type: ignore # noqa: PGH003
        msg = f"No ensemble generator is registered for type {type(head)}"
        raise NotImplementedError(msg)
    return subensemble_generator(base, num_heads=num_heads, head=head, reset_params=reset_params, head_layer=head_layer)
