"""Shared ensemble implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from flax.nnx import rnglib

    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor


@lazydispatch
def ensemble_generator[In, KwIn, Out](base: Predictor[In, KwIn, Out]) -> Predictor[In, KwIn, Out]:
    """Generate an ensemble from a base model."""
    msg = f"No ensemble generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


def register(cls: LazyType, generator: Callable) -> None:
    """Register a class which can be used as a base for an ensemble."""
    ensemble_generator.register(cls=cls, func=generator)


def ensemble[T: Predictor](
    base: T,
    num_members: int,
    reset_params: bool = True,
    seed: int = 1,
    rngs: rnglib.Rngs | None = None,
) -> T:
    """Create an ensemble predictor from a base predictor based on :cite:`lakshminarayananSimpleScalable2017`.

    Args:
        base: Predictor, The base model to be used for the ensemble.
        num_members: The number of members in the ensemble.
        reset_params: Whether to reset the parameters of each member.
        seed: int, seed to be used for deterministic member reset.
        rngs: nnx.Rngs used for flax member re-initialization, overwrites seed.

    Returns:
        Predictor, The ensemble predictor.
    """
    return ensemble_generator(base, num_members=num_members, reset_params=reset_params, seed=seed, rngs=rngs)
