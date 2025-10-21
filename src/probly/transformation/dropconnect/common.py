"""Shared DropConnect implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.traverse_nn import is_first_layer, nn_compose
from pytraverse import CLONE, GlobalVariable, lazydispatch_traverser, traverse

if TYPE_CHECKING:
    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser

P = GlobalVariable[float]("P", "The probability of dropconnect.")

dropconnect_traverser = lazydispatch_traverser[object](name="dropconnect_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be replaced by DropConnect layers."""
    dropconnect_traverser.register(cls=cls, traverser=traverser, skip_if=is_first_layer, vars={"p": P})


def dropconnect[T: Predictor](base: T, p: float = 0.25) -> T:
    """Create a DropConnect predictor from a base predictor.

    Args:
        base: The base model to be used for dropout.
        p: The probability of dropping out a neuron. Default is 0.25.

    Returns:
        The DropConnect predictor.
    """
    return traverse(base, nn_compose(dropconnect_traverser), init={P: p, CLONE: True})
