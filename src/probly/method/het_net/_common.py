"""Shared HetNets method implementation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from probly.predictor import LogitClassifier, Predictor, ProbabilisticClassifier, RandomPredictor, predict, predict_raw
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.entropy import LabelNoiseEntropyDecomposition
from probly.representation.distribution import CategoricalDistribution, create_categorical_distribution_from_logits
from probly.representer import Representer, representer
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse


class HetNetRepresentation[T](CategoricalDistribution[T]):
    """The categorical distribution predicted by a HetNets model.

    HetNets only capture aleatoric uncertainty, so the method-local registration
    routes this representation to an aleatoric-only decomposition.
    """


@runtime_checkable
class HetNetPredictor[**In, Out: CategoricalDistribution](RandomPredictor[In, Out], Protocol):
    """A predictor with a heteroscedastic classification head."""


het_net_traverser = flexdispatch_traverser[object](name="het_nets_traverser")

LAST_LAYER = GlobalVariable[bool]("LAST_LAYER")
NUM_FACTORS = GlobalVariable[int]("NUM_FACTORS")
TEMPERATURE = GlobalVariable[float]("TEMPERATURE")
IS_PARAMETER_EFFICIENT = GlobalVariable[bool]("IS_PARAMETER_EFFICIENT")
NUM_SAMPLES = GlobalVariable[int]("NUM_SAMPLES")


@predictor_transformation(
    permitted_predictor_types=(
        LogitClassifier,
        ProbabilisticClassifier,
    ),
    preserve_predictor_type=False,
)
@HetNetPredictor.register_factory
def het_net[**In, Out: CategoricalDistribution](
    base: Predictor[In, Out],
    num_factors: int = 10,
    temperature: float = 1.0,
    is_parameter_efficient: bool = False,
    num_samples: int = 100,
) -> HetNetPredictor[In, Out]:
    """Create a HetNets predictor from a base predictor base on :cite:`collierCorrelatedInputDependent2021`.

    Args:
        base: The base model to be transformed.
        num_factors: The rank of the low-rank covariance parametrization.
        temperature: The temperature parameter for scaling the utility.
        is_parameter_efficient: Whether to use the parameter-efficient version.
        num_samples: The number of Monte Carlo samples of the utility that the heteroscedastic
            layer draws and averages in every forward pass.

    Returns:
        The HetNets predictor.
    """
    return traverse(
        base,
        nn_compose(het_net_traverser),
        init={
            CLONE: True,
            LAST_LAYER: True,
            TRAVERSE_REVERSED: True,
            NUM_FACTORS: num_factors,
            TEMPERATURE: temperature,
            IS_PARAMETER_EFFICIENT: is_parameter_efficient,
            NUM_SAMPLES: num_samples,
        },
    )


@predict.register(HetNetPredictor)
def _[**In](
    predictor: HetNetPredictor[In, CategoricalDistribution],
    *args: In.args,
    **kwargs: In.kwargs,
) -> CategoricalDistribution:
    """Predict with a NetNets predictor."""
    return create_categorical_distribution_from_logits(predict_raw(predictor, *args, **kwargs))


class HetNetRepresenter[**In, Out: CategoricalDistribution](Representer[Any, In, Out, HetNetRepresentation]):
    """Representer that returns the prediction of a HetNets predictor.

    The heteroscedastic layer already averages over its Monte Carlo samples, so the
    predictor is called once.
    """

    def represent(self, *args: In.args, **kwargs: In.kwargs) -> HetNetRepresentation:
        """Predict the categorical distribution and mark it as a HetNets representation."""
        return HetNetRepresentation.register_instance(predict(self.predictor, *args, **kwargs))


representer.register(HetNetPredictor, HetNetRepresenter)
decompose.register(HetNetRepresentation, LabelNoiseEntropyDecomposition)
