"""Shared HetNets representer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from probly.method.het_nets import HetNetsPredictor
from probly.representation.het_nets import HetNetsRepresentation, create_het_nets_representation
from probly.representer._representer import representer
from probly.representer.sampler import Sampler

if TYPE_CHECKING:
    from probly.representer.sampler._common import SamplingStrategy


@representer.register(HetNetsPredictor)
class HetNetsRepresenter[**In](Sampler[In, Any, HetNetsRepresentation]):
    """A representer that draws samples from a HetNets predictor.

    Each call to the underlying predictor produces a single categorical distribution
    drawn by sampling once from the heteroscedastic latent utility model. Repeated
    calls form a Monte Carlo sample of categorical distributions.
    """

    def __init__(
        self,
        predictor: HetNetsPredictor[In, Any],
        num_samples: int,
        sampling_strategy: SamplingStrategy = "sequential",
    ) -> None:
        """Initialize the HetNets representer.

        Args:
            predictor: The HetNets predictor to be sampled from.
            num_samples: The number of samples to draw.
            sampling_strategy: How the samples should be computed.
        """
        super().__init__(
            predictor,
            num_samples=num_samples,
            sampling_strategy=sampling_strategy,
            sample_factory=create_het_nets_representation,
            sample_axis=0,
        )
