"""Shared HetNets implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.method.method import predictor_transformation
from probly.predictor import LogitDistributionPredictor, Predictor
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
)
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse


@runtime_checkable
class HetNetsPredictor[**In, Out: TorchCategoricalDistribution](LogitDistributionPredictor[In, Out], Protocol):
    """A predictor that applies HetNets."""


het_nets_traverser = flexdispatch_traverser[object](name="het_nets_traverser")

LAST_LAYER = GlobalVariable[bool]("LAST_LAYER")
NUM_FACTORS = GlobalVariable[int]("NUM_FACTORS")
TEMPERATURE = GlobalVariable[float]("TEMPERATURE")
NUM_MC_SAMPLES = GlobalVariable[int]("NUM_MC_SAMPLES")
IS_PARAMETER_EFFICIENT = GlobalVariable[bool]("IS_PARAMETER_EFFICIENT")
MULTILABEL = GlobalVariable[bool]("MULTILABEL")


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)  # ty:ignore[invalid-argument-type]
@HetNetsPredictor.register_factory
def het_nets[**In, Out: TorchCategoricalDistribution](
    base: Predictor[In, Out],
    num_factors: int = 10,
    temperature: float = 1.0,
    num_mc_samples: int = 10,
    is_parameter_efficient: bool = False,
    multilabel: bool = False,
) -> HetNetsPredictor[In, Out]:
    """Create a HetNets predictor from a base predictor base on :cite:`collier2021hetnets`.

    Args:
        base: The base model to be used for HetNets.
        num_factors: The rank of the low-rank covariance parametrization. Default is 10.
        temperature: The temperature parameter for scaling the utility. Default is 1.0.
        num_mc_samples: The number of Monte Carlo samples to use during training. Default is 10.
        is_parameter_efficient: Whether to use the parameter-efficient version of HetNets. Default is False.
        multilabel: Whether the task is multilabel. Default is False.

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
            NUM_MC_SAMPLES: num_mc_samples,
            IS_PARAMETER_EFFICIENT: is_parameter_efficient,
            MULTILABEL: multilabel,
        },
    )
