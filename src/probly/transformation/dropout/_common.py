"""Shared dropout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from probly.predictor import RandomPredictor
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import is_first_layer, nn_compose
from pytraverse import CLONE, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from flax.nnx.rnglib import Rngs, RngStream

    from flextype.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser


@runtime_checkable
class DropoutPredictor[**In, Out](RandomPredictor[In, Out], Protocol):
    """A predictor that applies dropout."""


P = GlobalVariable[float]("P", "The probability of dropout.")

type RNG = int | Rngs | RngStream

RNGS = GlobalVariable[RNG]("RNGS", "rngs for flax layer initialization.")
RNG_COLLECTION = GlobalVariable[str]("RNG_COLLECTION", "rng_collection for flax layer initialization")
SHARED_MASK = GlobalVariable[bool]("SHARED_MASK", "Insert shared-mask dropout layers instead of standard dropout.")

dropout_traverser = flexdispatch_traverser[object](name="dropout_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be prepended by Dropout layers."""
    dropout_traverser.register(
        cls=cls,
        traverser=traverser,
        skip_if=is_first_layer,
        vars={"p": P, "rng_collection": RNG_COLLECTION, "rngs": RNGS, "shared_mask": SHARED_MASK},
    )


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=True)
@DropoutPredictor.register_factory
def dropout[T: Predictor](
    base: T,
    p: float = 0.25,
    rng_collection: str = "dropout",
    rngs: Rngs | RngStream | int = 1,
    shared_mask: bool = False,
) -> T:
    """Create a Dropout predictor from a base predictor based on :cite:`galDropoutBayesian2016`.

    Args:
        base: The base model to be used for dropout.
        p: The probability of dropping out a neuron. Default is 0.25.
        rng_collection: Optional rng collection name for flax layer initialization. Default is "dropout".
        rngs: Optional rngs for flax layer initialization. Default is 1.
        shared_mask: If True, insert shared-mask dropout layers that draw one mask per forward
            pass shared across the batch (torch backend only). Shared masking applies only to the
            dropout layers this transform inserts; any pre-existing dropout keeps its standard
            per-element masks. Default is False.

    Returns:
        The DropOut predictor.
    """
    if p < 0 or p > 1:
        msg = f"The probability p must be between 0 and 1, but got {p} instead."
        raise ValueError(msg)
    return traverse(
        base,
        nn_compose(dropout_traverser),
        init={P: p, CLONE: True, RNG_COLLECTION: rng_collection, RNGS: rngs, SHARED_MASK: shared_mask},
    )
