"""HetNets method compatibility layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

from probly.representation.distribution import CategoricalDistribution
from probly.representation.het_nets import HetNetsRepresentation
from probly.representation.sample import create_sample
from probly.representer import representer
from probly.representer.sampler import Sampler
from probly.transformation.heteroscedastic_classification import (
    HeteroscedasticClassificationPredictor,
    heteroscedastic_classification,
    heteroscedastic_classification_traverser,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@runtime_checkable
class HetNetsPredictor[**In, Out: CategoricalDistribution](HeteroscedasticClassificationPredictor[In, Out], Protocol):
    """A predictor routed through the HetNets representer."""


het_nets = HetNetsPredictor.register_factory(heteroscedastic_classification)
het_nets_traverser = heteroscedastic_classification_traverser


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


__all__ = ["HetNetsPredictor", "create_het_nets_sample", "het_nets", "het_nets_traverser"]
