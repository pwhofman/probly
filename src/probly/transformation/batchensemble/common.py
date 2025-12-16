"""Shared subensemble implementation."""

from __future__ import annotations

import math
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

import math

   
NUM_MEMBERS = GlobalVariable[bool]("USE_BASE_WEIGHTS", default=1)
S_MEAN = GlobalVariable[float]("S_MEAN", default=1.0)
S_STD = GlobalVariable[float]("S_STD", default=0.01)
R_MEAN = GlobalVariable[float]("R_MEAN", default=1.0)
R_STD = GlobalVariable[float]("R_STD", default=0.01)
KAIMING_SLOPE = GlobalVariable[float]("KAIMING_SLOPE", default=math.sqrt(5))

batchensemble_traverser = lazydispatch_traverser[object](name="batchensemble_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be replaced by Batchensemble layers."""
    batchensemble_traverser.register(
        cls=cls,
        traverser=traverser,
        vars={
            "num_members": NUM_MEMBERS,
            "s_mean": S_MEAN,
            "s_std": S_STD,
            "r_mean": R_MEAN,
            "r_std": R_STD,
            "kaiming_slope": KAIMING_SLOPE,
        },
    )

def batchensemble[T: Predictor](
    base: T,
    num_members: int,
    s_mean: float = S_MEAN.default,
    s_std: float = S_STD.default,
    r_mean: float = R_MEAN.default,
    r_std: float = R_STD.default,
    kaiming_slope: float = KAIMING_SLOPE.default,
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
    if not s_std > 0:
        msg = (
            "The initial standard deviation of the input modulation s must be greater than 0, "
            f"but got {s_std} instead."
        )
        raise ValueError(msg)
    if not r_std > 0:
        msg = (
            "The initial standard deviation of the output modulation r must be greater than 0, "
            f"but got {r_std} instead."
        )
        raise ValueError(msg)
    # TODO maybe check that the mean of r and s should be greater than zero. 
    # some constraints on the kaiming slope?
    
    return traverse(
        base,
        nn_compose(batchensemble_traverser),
        init={
            NUM_MEMBERS: num_members,
            S_MEAN: s_mean,
            S_STD: s_std,
            R_MEAN: r_mean,
            R_STD: r_std,
            KAIMING_SLOPE: kaiming_slope,
            CLONE: True,
        },
    )
