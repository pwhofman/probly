"""Shared HetNets implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from probly.method.method import predictor_transformation
from probly.predictor import RandomPredictor
from probly.traverse_nn import nn_compose
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, lazydispatch_traverser, traverse

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser


@runtime_checkable
class HetNetsPredictor[**In, Out](RandomPredictor[In, Out], Protocol):
    """A predictor that applies HetNets."""


hetnets_traverser = lazydispatch_traverser[object](name="hetnets_traverser")

LAST_LAYER = GlobalVariable[bool]("LAST_LAYER")
NUM_FACTORS = GlobalVariable[int]("NUM_FACTORS")
TEMPERATURE = GlobalVariable[float]("TEMPERATURE")
NUM_MC_SAMPLES = GlobalVariable[int]("NUM_SAMPLES")
IS_PARAMETER_EFFICIENT = GlobalVariable[bool]("IS_PARAMETER_EFFICIENT")
MULTILABEL = GlobalVariable[bool]("MULTILABEL")


def register(cls: type, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be transformed by HetNets."""
    hetnets_traverser.register(cls=cls, traverser=traverser, vars={"last_layer": LAST_LAYER})


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=True)
@HetNetsPredictor.register_factory
def het_nets[T: Predictor](
    base: T,
    num_factors: int = 10,
    temperature: float = 1.0,
    num_mc_samples: int = 10,
    is_parameter_efficient: bool = False,
    multilabel: bool = False,
) -> T:
    """Create a HetNets predictor from a base predictor.

    Args:
        base: Predictor, The base model to be used for HetNets.
        num_factors: int, The rank of the low-rank covariance parametrization. Default is 10.
        temperature: float, The temperature parameter for scaling the utility. Default is 1.0.
        num_mc_samples: int, The number of Monte Carlo samples to use during training. Default is 10.
        is_parameter_efficient: bool, Whether to use the parameter-efficient version of HetNets. Default is False.
        multilabel: bool, Whether the task is multilabel. Default is False.

    Returns:
        Predictor, The HetNets predictor.
    """
    return traverse(
        base,
        nn_compose(hetnets_traverser),
        init={
            LAST_LAYER: True,
            TRAVERSE_REVERSED: True,
            NUM_FACTORS: num_factors,
            TEMPERATURE: temperature,
            NUM_MC_SAMPLES: num_mc_samples,
            IS_PARAMETER_EFFICIENT: is_parameter_efficient,
            MULTILABEL: multilabel,
        },
    )
