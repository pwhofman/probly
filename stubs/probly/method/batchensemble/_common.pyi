"""Shared BatchEnsemble implementation."""

from __future__ import annotations
import probly

from typing import TYPE_CHECKING

from probly.method.method import predictor_transformation
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, GlobalVariable, lazydispatch_traverser, traverse

if TYPE_CHECKING:
    from flax.nnx.rnglib import Rngs

    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser


NUM_MEMBERS = GlobalVariable[int]("NUM_MEMBERS", default=1)
USE_BASE_WEIGHTS = GlobalVariable[bool]("USE_BASE_WEIGHT", default=False)
S_MEAN = GlobalVariable[float]("S_MEAN", default=1.0)
S_STD = GlobalVariable[float]("S_STD", default=0.01)
R_MEAN = GlobalVariable[float]("R_MEAN", default=1.0)
R_STD = GlobalVariable[float]("R_STD", default=0.01)
type RNG = Rngs | int
RNGS = GlobalVariable[RNG]("RNGS", "rngs for flax layer initialization.", default=1)

batchensemble_traverser = lazydispatch_traverser[object](name="batchensemble_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be replaced by Batchensemble layers."""
    ...
def batchensemble[T: Predictor](base: T, num_members: int = NUM_MEMBERS.default, use_base_weights: bool = USE_BASE_WEIGHTS.default, s_mean: float = S_MEAN.default, s_std: float = S_STD.default, r_mean: float = R_MEAN.default, r_std: float = R_STD.default, rngs: Rngs | int = RNGS.default, *, predictor_type: probly.predictor.PredictorName | type[probly.predictor.Predictor] | None = None) -> T:
    """Create a Batchensemble predictor from a base predictor.

    It calls a traverser to replace all linear and convolutional layers by their BatchEnsemble
    counterparts.

    Args:
        base: Predictor, The model in which the layers will be replaced by BatchEnsemble layers.
        num_members: int, The number of members in the BatchEnsemble.
        use_base_weights: bool, Whether to use the weights of the base layer as prior means.
        s_mean: float, The mean of the input modulation s, drawn from `nn.init._normal(s_mean, s_std)`.
        s_std: float, The standard deviation of the input modulation s, drawn from `nn.init._normal(s_mean, s_std)`.
        r_mean: float, The mean of the output modulation r, drawn from `nn.init._normal(r_mean, r_std)`.
        r_std: float, The standard deviation of the output modulation r, drawn from `nn.init._normal(r_mean, r_std)`.
        rngs: nnx.Rngs | int, The rngs used for flax layer initialization.

    Returns:
        Predictor, The BatchEnsemble predictor.

    Raises:
        ValueError: If `num_members` is not a positive integer.
        ValueError: If `s_std` is not greater than 0.
        ValueError: If `s_mean` is not greater than 0.
        ValueError: If `r_std` is not greater than 0.
        ValueError: If `r_mean` is not greater than 0.
    """
    ...
