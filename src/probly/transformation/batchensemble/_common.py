"""Shared BatchEnsemble implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.predictor import Predictor
from probly.representer._representer import DummyRepresenter, representer
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from flax.nnx.rnglib import Rngs

    from flextype.isinstance import LazyType
    from pytraverse.composition import RegisteredLooseTraverser


@runtime_checkable
class BatchEnsemblePredictor[**In, Out](Predictor[In, Out], Protocol):
    """Protocol marking a predictor whose linear/conv layers were swapped for BatchEnsemble layers."""


representer.register(BatchEnsemblePredictor, DummyRepresenter)

type InitMethod = Literal["random_sign", "normal"]

NUM_MEMBERS = GlobalVariable[int]("NUM_MEMBERS", default=1)
USE_BASE_WEIGHTS = GlobalVariable[bool]("USE_BASE_WEIGHT", default=False)
INIT = GlobalVariable[InitMethod]("INIT", default="normal")
R_MEAN = GlobalVariable[float]("R_MEAN", default=1.0)
R_STD = GlobalVariable[float]("R_STD", default=0.5)
S_MEAN = GlobalVariable[float]("S_MEAN", default=1.0)
S_STD = GlobalVariable[float]("S_STD", default=0.5)
type RNG = Rngs | int
RNGS = GlobalVariable[RNG]("RNGS", "rngs for flax layer initialization.", default=1)

batchensemble_traverser = flexdispatch_traverser[object](name="batchensemble_traverser")


@flexdispatch
def _attach_num_members(model: object, num_members: int) -> None:
    """Attach the ensemble size to the traversed model so it survives serialization."""
    msg = f"No num_members attacher registered for type {type(model)}."
    raise NotImplementedError(msg)


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be replaced by Batchensemble layers."""
    batchensemble_traverser.register(
        cls=cls,
        traverser=traverser,
        vars={
            "num_members": NUM_MEMBERS,
            "use_base_weights": USE_BASE_WEIGHTS,
            "init": INIT,
            "r_mean": R_MEAN,
            "r_std": R_STD,
            "s_mean": S_MEAN,
            "s_std": S_STD,
            "rngs": RNGS,
        },
    )


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@BatchEnsemblePredictor.register_factory
def batchensemble[**In, Out](
    base: Predictor[In, Out],
    num_members: int = NUM_MEMBERS.default,
    use_base_weights: bool = USE_BASE_WEIGHTS.default,
    init: InitMethod = INIT.default,
    r_mean: float = R_MEAN.default,
    r_std: float = R_STD.default,
    s_mean: float = S_MEAN.default,
    s_std: float = S_STD.default,
    rngs: Rngs | int = RNGS.default,
) -> BatchEnsemblePredictor[In, Out]:
    """Create a BatchEnsemble predictor from a base predictor based on :cite:`wenBatchEnsemble2020`.

    Replaces all linear and convolutional layers with their BatchEnsemble counterparts and tags
    the result with ``num_members`` so :func:`predict` can tile inputs and wrap outputs as a
    Sample.

    Args:
        base: The model in which the layers will be replaced by BatchEnsemble layers.
        num_members: The number of members in the BatchEnsemble.
        use_base_weights: Whether to use the weights of the base layer as initial weights.
        init: Initialization scheme for ``r`` and ``s`` - ``"normal"`` (Gaussian, imagenet
            baseline default) or ``"random_sign"`` ({-1, +1}, paper Appendix B).
        r_mean: mean of the Gaussian initializer for ``r`` when ``init="normal"``.
        r_std: standard deviation of the Gaussian initializer for ``r`` when ``init="normal"``.
        s_mean: mean of the Gaussian initializer for ``s`` when ``init="normal"``.
        s_std: standard deviation of the Gaussian initializer for ``s`` when ``init="normal"``.
        rngs: The rngs used for flax layer initialization.

    Returns:
        The BatchEnsemble predictor.

    Raises:
        ValueError: If `num_members` is not a positive integer.
        ValueError: If `init` is not ``"normal"`` or ``"random_sign"``.
        ValueError: If `r_std` or `s_std` is not strictly positive when ``init="normal"``.
    """
    if num_members < 1:
        msg = f"num_members must be a positive integer, got {num_members}."
        raise ValueError(msg)
    if init not in ("normal", "random_sign"):
        msg = f"init must be 'normal' or 'random_sign', got {init!r}."
        raise ValueError(msg)
    if init == "normal":
        if not r_std > 0:
            msg = f"r_std must be greater than 0 when init='normal', got {r_std}."
            raise ValueError(msg)
        if not s_std > 0:
            msg = f"s_std must be greater than 0 when init='normal', got {s_std}."
            raise ValueError(msg)

    transformed = traverse(
        base,
        nn_compose(batchensemble_traverser),
        init={
            NUM_MEMBERS: num_members,
            USE_BASE_WEIGHTS: use_base_weights,
            INIT: init,
            R_MEAN: r_mean,
            R_STD: r_std,
            S_MEAN: s_mean,
            S_STD: s_std,
            RNGS: rngs,
            CLONE: True,
        },
    )
    # Tag the model so downstream code (predict, train_funcs) can recover the ensemble size.
    _attach_num_members(transformed, num_members)
    return transformed
