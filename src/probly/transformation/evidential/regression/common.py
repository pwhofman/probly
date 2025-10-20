"""Shared evidential regression implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, lazy_singledispatch_traverser, traverse

if TYPE_CHECKING:
    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser

REPLACED_LAST_LINEAR = GlobalVariable[bool](
    "REPLACED_LAST_LINEAR",
    "Whether the last linear layer has been replaced with a NormalInverseGammaLinear layer.",
    default=False,
)

evidential_regression_traverser = lazy_singledispatch_traverser[object](name="evidential_regression_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be replaced by a normal inverse gamma layer."""
    evidential_regression_traverser.register(cls=cls, traverser=traverser, skip_if=lambda s: s[REPLACED_LAST_LINEAR])


def evidential_regression[In, KwIn, Out](base: Predictor[In, KwIn, Out]) -> Predictor[In, KwIn, Out]:
    """Create an evidential regression predictor from a base predictor.

    Args:
        base: Predictor, The base model to be used for evidential regression.

    Returns:
        Predictor, The evidential regression predictor.
    """
    return traverse(base, nn_compose(evidential_regression_traverser), init={TRAVERSE_REVERSED: True, CLONE: True})
