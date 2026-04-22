"""Shared Bayesian implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from probly.method.method import predictor_transformation
from probly.predictor import RandomPredictor
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from flextype.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser


class BayesianPredictor[**In, Out](RandomPredictor[In, Out], Protocol):
    """A predictor that applies Bayesian layers."""


USE_BASE_WEIGHTS = GlobalVariable[bool]("USE_BASE_WEIGHTS", default=False)
POSTERIOR_STD = GlobalVariable[float]("POSTERIOR_STD", default=0.05)
PRIOR_MEAN = GlobalVariable[float]("PRIOR_MEAN", default=0.0)
PRIOR_STD = GlobalVariable[float]("PRIOR_STD", default=1.0)

bayesian_traverser = flexdispatch_traverser[object](name="bayesian_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be replaced by Bayesian layers."""
    bayesian_traverser.register(
        cls=cls,
        traverser=traverser,
        vars={
            "use_base_weights": USE_BASE_WEIGHTS,
            "posterior_std": POSTERIOR_STD,
            "prior_mean": PRIOR_MEAN,
            "prior_std": PRIOR_STD,
        },
    )


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=True)  # ty: ignore[invalid-argument-type]
@BayesianPredictor.register_factory
def bayesian[**In, Out](
    base: Predictor[In, Out],
    use_base_weights: bool = USE_BASE_WEIGHTS.default,
    posterior_std: float = POSTERIOR_STD.default,
    prior_mean: float = PRIOR_MEAN.default,
    prior_std: float = PRIOR_STD.default,
) -> BayesianPredictor[In, Out]:
    """Create a Bayesian predictor from a base predictor based on :cite:`blundellWeightUncertainty2015`.

    Args:
        base: The base model to be used for the Bayesian neural network.
        use_base_weights: bool, If True, the weights of the base model are used as the prior mean.
        posterior_std: float, The initial posterior standard deviation.
        prior_mean: float, The prior mean.
        prior_std: float, The prior standard deviation.

    Returns:
        The Bayesian predictor.
    """
    if posterior_std <= 0:
        msg = (
            "The initial posterior standard deviation posterior_std must be greater than 0, "
            f"but got {posterior_std} instead."
        )
        raise ValueError(msg)
    if prior_std <= 0:
        msg = f"The prior standard deviation prior_std must be greater than 0, but got {prior_std} instead."
        raise ValueError(msg)
    return traverse(
        base,
        nn_compose(bayesian_traverser),
        init={
            USE_BASE_WEIGHTS: use_base_weights,
            POSTERIOR_STD: posterior_std,
            PRIOR_MEAN: prior_mean,
            PRIOR_STD: prior_std,
            CLONE: True,
        },
    )
