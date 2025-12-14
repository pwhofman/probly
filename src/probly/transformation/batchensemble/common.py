"""Shared subensemble implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_dispatch import lazydispatch
from probly.transformation.subensemble.common import subensemble_generator
from __future__ import annotations

from typing import TYPE_CHECKING

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, GlobalVariable, lazydispatch_traverser, traverse

if TYPE_CHECKING:
    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser


   
NUM_MEMBERS = GlobalVariable[bool]("USE_BASE_WEIGHTS", default=1)

batchensemble_traverser = lazydispatch_traverser[object](name="batchensemble_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be replaced by Batchensemble layers."""
    batchensemble_traverser.register(
        cls=cls,
        traverser=traverser,
        vars={
            "num_members": NUM_MEMBERS,
        },
    )

def batchensemble[T: Predictor](
    base: T,
    num_members: int,
) -> T:
    """Create a Batchensemble predictor from a base predictor.

    Args:
        base: Predictor, The model to be used as backbone or to create the backbone and heads.
        num_members: int, The number of members in the batchensemble.

    Returns:
        Predictor, The batchensemble predictor.

    Raises:
        ValueError: If `num_members` is not a positive integer.
    """
    if num_members < 1:
        msg = f"num_members must be a positive integer, got {num_members}."
        raise ValueError(msg)
    return traverse(
        base,
        nn_compose(batchensemble_traverser),
        init={
            NUM_MEMBERS: num_members,
            CLONE: True,
        },
    )
