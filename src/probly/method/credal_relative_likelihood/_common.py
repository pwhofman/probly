"""Shared credal relative likelihood implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from probly.method.ensemble import EnsemblePredictor, register_ensemble_members
from probly.method.method import predictor_transformation
from probly.predictor import ProbabilisticClassifier
from probly.traverse_nn import nn_compose, reset_traverser
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from probly.predictor import Predictor


class CredalRelativeLikelihoodPredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor that applies the credal relative likelihood transformation."""


INITIALIZED = GlobalVariable[bool]("INITIALIZED", default=False)
RESET_PARAMS = GlobalVariable[bool]("RESET_PARAMS", default=True)
BIAS_CLS = GlobalVariable[int]("BIAS_CLS", default=0)
TOBIAS_VALUE = GlobalVariable[int]("TOBIAS_VALUE", default=100)

credal_relative_likelihood_traverser = flexdispatch_traverser[object](name="credal_relative_likelihood_traverser")


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier,), post_transform=register_ensemble_members
)
@CredalRelativeLikelihoodPredictor.register_factory
def credal_relative_likelihood[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True, tobias_value: int = 100
) -> CredalRelativeLikelihoodPredictor[In, Out]:
    """Create a credal relative likelihood predictor from a base predictor based on :cite:`lohr2025credal`.

    Args:
        base: The base model to be used for the credal relative likelihood ensemble.
        num_members: The number of members in the credal relative likelihood ensemble.
        reset_params: Whether to reset the parameters of each member.
        tobias_value: The value to use for the credal relative likelihood initialization.

    Returns:
        The credal relative likelihood ensemble predictor.
    """
    if reset_params:
        traverser = nn_compose(reset_traverser, credal_relative_likelihood_traverser)
    else:
        traverser = nn_compose(credal_relative_likelihood_traverser)
    members = [
        traverse(
            base,
            traverser,
            init={
                BIAS_CLS: i,
                TOBIAS_VALUE: tobias_value,
                INITIALIZED: False,
                RESET_PARAMS: reset_params,
                TRAVERSE_REVERSED: True,
            },
        )
        for i in range(num_members)
    ]
    return members  # ty:ignore[invalid-return-type]
