"""Shared dropout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.traverse_nn import is_first_layer, nn_compose
from pytraverse import CLONE, GlobalVariable, lazy_singledispatch_traverser, traverse

if TYPE_CHECKING:
    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser

P = GlobalVariable[float]("P", "The probability of dropout.")

dropout_traverser = lazy_singledispatch_traverser[object](name="dropout_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be prepended by Dropout layers."""
    dropout_traverser.register(cls=cls, traverser=traverser, skip_if=is_first_layer, vars={"p": P})


def dropout[In, KwIn, Out](base: Predictor[In, KwIn, Out], p: float = 0.25) -> Predictor[In, KwIn, Out]:
    """Create a Dropout ensemble predictor from a base predictor.

    Args:
        base: Predictor, The base model to be used for dropout.
        p: float, The probability of dropping out a neuron.  Default is 0.25.

    Returns:
        Predictor, The dropout ensemble predictor.
    """
    return traverse(base, nn_compose(dropout_traverser), init={P: p, CLONE: True})
