"""Shared HetNets method implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

from probly.predictor import LogitClassifier, Predictor, ProbabilisticClassifier, RandomPredictor, predict, predict_raw
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.entropy import LabelNoiseEntropyDecomposition
from probly.representation.distribution import CategoricalDistribution, create_categorical_distribution_from_logits
from probly.representation.distribution._common import CategoricalDistributionSample
from probly.representation.sample import create_sample
from probly.representer import representer
from probly.representer.sampler import Sampler
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from collections.abc import Iterable


class HetNetsRepresentation[T: CategoricalDistribution](CategoricalDistributionSample[T]):
    """A sample of categorical distributions produced by a HetNets model.

    HetNets only capture aleatoric uncertainty, so the method-local registration
    routes this representation to an aleatoric-only decomposition.
    """


@runtime_checkable
class HetNetsPredictor[**In, Out: CategoricalDistribution](RandomPredictor[In, Out], Protocol):
    """A predictor with a heteroscedastic classification head."""


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
        base: The base model to be transformed.
        num_factors: The rank of the low-rank covariance parametrization.
        temperature: The temperature parameter for scaling the utility.
        is_parameter_efficient: Whether to use the parameter-efficient version.

    Returns:
        The HetNets predictor.
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
    predictor: HetNetsPredictor[In, CategoricalDistribution],
    *args: In.args,
    **kwargs: In.kwargs,
) -> CategoricalDistribution:
    """Predict with a NetNets predictor."""
    return create_categorical_distribution_from_logits(predict_raw(predictor, *args, **kwargs))


def create_het_nets_sample[Out: CategoricalDistribution](
    predictions: Iterable[Out], sample_axis: int = -1
) -> HetNetsRepresentation:
    """Create a HetNets sample representation from repeated predictions.

    Args:
        predictions: The repeated predictions to collect into a sample.
        sample_axis: The axis along which samples are organized.

    Returns:
        The created sample marked as a HetNets representation.
    """
    return cast(
        "HetNetsRepresentation",
        HetNetsRepresentation.register_instance(create_sample(predictions, sample_axis=sample_axis)),
    )


class HetNetsRepresenter[**In, Out: CategoricalDistribution](Sampler[In, Out, HetNetsRepresentation]):
    """Representer that draws samples from a HetNets predictor."""

    def __init__(
        self,
        predictor: HetNetsPredictor[In, Out],
        num_samples: int,
        sampling_strategy: Literal["sequential"] = "sequential",
        sample_axis: int = -1,
    ) -> None:
        """Initialize the HetNets representer.

        Args:
            predictor: The predictor to sample from.
            num_samples: Number of Monte Carlo samples to draw.
            sampling_strategy: How repeated predictions should be computed.
            sample_axis: Axis along which samples are organized.
        """
        super().__init__(predictor, num_samples, sampling_strategy, create_het_nets_sample, sample_axis)


representer.register(HetNetsPredictor, HetNetsRepresenter)
decompose.register(HetNetsRepresentation, LabelNoiseEntropyDecomposition)
