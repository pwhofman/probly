"""Shared dropout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.traverse_nn import is_first_layer, nn_compose
from pytraverse import CLONE, GlobalVariable, lazydispatch_traverser, traverse

if TYPE_CHECKING:
    from flax.nnx.rnglib import Rngs, RngStream

    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser

P = GlobalVariable[float]("P", "The probability of dropout.")

type RNG = int | Rngs | RngStream

RNGS = GlobalVariable[RNG]("RNGS", "rngs for flax layer initialization.")
RNG_COLLECTION = GlobalVariable[str]("RNG_COLLECTION", "rng_collection for flax layer initialization")

dropout_traverser = lazydispatch_traverser[object](name="dropout_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be prepended by Dropout layers."""
    dropout_traverser.register(
        cls=cls,
        traverser=traverser,
        skip_if=is_first_layer,
        vars={"p": P, "rng_collection": RNG_COLLECTION, "rngs": RNGS},
    )


def dropout[T: Predictor](
    base: T, p: float = 0.25, rng_collection: str = "dropout", rngs: Rngs | RngStream | int = 1
) -> T:
    """Create a Dropout predictor from a base predictor based on :cite:`galDropoutBayesian2016`.

    Args:
        base: Predictor, The base model to be used for dropout.
        p: float, The probability of dropping out a neuron.  Default is 0.25.
        rng_collection: Optional str for flax layer initialization. Default is "dropout".
        rngs: Optional rngs for flax layer initialization (types: rnglib.Rngs | rnglib.RngStream | int), default: 1.

    Returns:
        Predictor, The DropOut predictor.
    """
    if p < 0 or p > 1:
        msg = f"The probability p must be between 0 and 1, but got {p} instead."
        raise ValueError(msg)
    return traverse(
        base, nn_compose(dropout_traverser), init={P: p, CLONE: True, RNG_COLLECTION: rng_collection, RNGS: rngs}
    )
