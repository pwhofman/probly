"""Shared HetNets implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.predictor import LogitClassifier, Predictor, ProbabilisticClassifier, RandomPredictor, predict, predict_raw
from probly.representation.distribution import (
    CategoricalDistribution,
    create_categorical_distribution_from_logits,
)
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse


@runtime_checkable
class HetNetsPredictor[**In, Out: CategoricalDistribution](RandomPredictor[In, Out], Protocol):
    """A predictor that implements HetNets."""


het_nets_traverser = flexdispatch_traverser[object](name="het_nets_traverser")

LAST_LAYER = GlobalVariable[bool]("LAST_LAYER")
NUM_FACTORS = GlobalVariable[int]("NUM_FACTORS")
TEMPERATURE = GlobalVariable[float]("TEMPERATURE")
IS_PARAMETER_EFFICIENT = GlobalVariable[bool]("IS_PARAMETER_EFFICIENT")


@predictor_transformation(
    permitted_predictor_types=(
        LogitClassifier,
        ProbabilisticClassifier,
    ),
    preserve_predictor_type=False,
)  # ty:ignore[invalid-argument-type]
@HetNetsPredictor.register_factory
def het_nets[**In, Out: CategoricalDistribution](
    base: Predictor[In, Out],
    num_factors: int = 10,
    temperature: float = 1.0,
    is_parameter_efficient: bool = False,
) -> HetNetsPredictor[In, Out]:
    """Create a HetNets predictor from a base predictor base on :cite:`collier2021hetnets`.

    Args:
        base: The base model to be used for HetNets.
        num_factors: The rank of the low-rank covariance parametrization. Default is 10.
        temperature: The temperature parameter for scaling the utility. Default is 1.0.
        is_parameter_efficient: Whether to use the parameter-efficient version of HetNets. Default is False.

    Returns:
        Predictor, The HetNets predictor.
    """
    return traverse(
        base,
        nn_compose(het_nets_traverser),
        init={
            CLONE: True,
            LAST_LAYER: True,
            TRAVERSE_REVERSED: True,
            NUM_FACTORS: num_factors,
            TEMPERATURE: temperature,
            IS_PARAMETER_EFFICIENT: is_parameter_efficient,
        },
    )


@predict.register(HetNetsPredictor)
def _[**In](
    predictor: HetNetsPredictor[In, CategoricalDistribution], *args: In.args, **kwargs: In.kwargs
) -> CategoricalDistribution:
    """Predict with a HetNets predictor."""
    return create_categorical_distribution_from_logits(predict_raw(predictor, *args, **kwargs))
