"""Shared DropConnect implementation."""

from __future__ import annotations
import probly

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from probly.method.method import predictor_transformation
from probly.predictor import RandomPredictor
from probly.traverse_nn import is_first_layer, nn_compose
from pytraverse import CLONE, GlobalVariable, lazydispatch_traverser, traverse

if TYPE_CHECKING:
    from flax.nnx.rnglib import Rngs, RngStream

    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser


@runtime_checkable
class DropConnectPredictor[**In, Out](RandomPredictor[In, Out], Protocol):
    """A predictor that applies DropConnect."""


P = GlobalVariable[float]("P", "The probability of dropconnect.")

type RNG = int | Rngs | RngStream

RNGS = GlobalVariable[RNG]("RNGS", "rngs for flax layer initialization")
RNG_COLLECTION = GlobalVariable[str]("RNG_COLLECTION", "rng_collection for flax layer initialization")

dropconnect_traverser = lazydispatch_traverser[object](name="dropconnect_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be replaced by DropConnect layers."""
    ...
def dropconnect[T: Predictor](base: T, p: float = 0.25, rng_collection: str = 'dropconnect', rngs: Rngs | RngStream | int = 1, *, predictor_type: probly.predictor.PredictorName | type[probly.predictor.Predictor] | None = None) -> T:
    """Create a DropConnect predictor from a base predictor based on :cite:`mobinyDropConnectEffective2019`.

    Args:
        base: The base model to be used for dropout.
        p: The probability of dropping out a neuron. Default is 0.25.
        rng_collection: Optional str for flax layer initialization. Default is "dropconnect".
        rngs: Optional rngs for flax layer initialization (types: rnglib.Rngs | rnglib.RngStream | int), default: 1.

    Returns:
        The DropConnect predictor.
    """
    ...
