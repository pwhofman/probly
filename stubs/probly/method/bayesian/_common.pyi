"""Shared Bayesian implementation."""

from __future__ import annotations
import probly

from typing import TYPE_CHECKING, Protocol

from probly.method.method import predictor_transformation
from probly.predictor import RandomPredictor
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, GlobalVariable, lazydispatch_traverser, traverse

if TYPE_CHECKING:
    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser


class BayesianPredictor[**In, Out](RandomPredictor[In, Out], Protocol):
    """A predictor that applies Bayesian layers."""


USE_BASE_WEIGHTS = GlobalVariable[bool]("USE_BASE_WEIGHTS", default=False)
POSTERIOR_STD = GlobalVariable[float]("POSTERIOR_STD", default=0.05)
PRIOR_MEAN = GlobalVariable[float]("PRIOR_MEAN", default=0.0)
PRIOR_STD = GlobalVariable[float]("PRIOR_STD", default=1.0)

bayesian_traverser = lazydispatch_traverser[object](name="bayesian_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be replaced by Bayesian layers."""
    ...
def bayesian[**In, Out](base: Predictor[In, Out], use_base_weights: bool = USE_BASE_WEIGHTS.default, posterior_std: float = POSTERIOR_STD.default, prior_mean: float = PRIOR_MEAN.default, prior_std: float = PRIOR_STD.default, *, predictor_type: probly.predictor.PredictorName | type[probly.predictor.Predictor] | None = None) -> BayesianPredictor[In, Out]:
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
    ...
