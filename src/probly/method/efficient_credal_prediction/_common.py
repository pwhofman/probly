"""Shared efficient credal prediction implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.traverse_nn import nn_compose
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, lazydispatch_traverser, traverse

if TYPE_CHECKING:
    from lazy_dispatch import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser

efficient_credal_prediction_traverser = lazydispatch_traverser[object](name="efficient_credal_prediction_traverser")

LAST_LAYER = GlobalVariable[bool]("LAST_LAYER", default=True)


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be prepended by Dropout layers."""
    efficient_credal_prediction_traverser.register(cls=cls, traverser=traverser, vars={"last_layer": LAST_LAYER})


def efficient_credal_prediction[T: Predictor](base: T) -> T:
    """Create an efficient credal predictor from a base predictor based on :cite:`hofmanefficient`.

    Args:
        base: Predictor, The base model to be used for the efficient credal predictor.

    Returns:
        Predictor, The efficient credal predictor.
    """
    return traverse(
        base, nn_compose(efficient_credal_prediction_traverser), init={LAST_LAYER: True, TRAVERSE_REVERSED: True}
    )
