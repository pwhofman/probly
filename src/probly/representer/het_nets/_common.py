"""Shared HetNets representer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from probly.method.het_nets import HetNetsPredictor
from probly.representation.het_nets import HetNetsRepresentation, create_het_nets_representation
from probly.representer._representer import Representer, representer
from probly.representer.sampler import Sampler

if TYPE_CHECKING:
    from probly.representation.sample import Sample
    from probly.representer.sampler._common import SamplingStrategy


@representer.register(HetNetsPredictor)
class HetNetsRepresenter[**In, S: Sample](Representer[Any, In, HetNetsRepresentation, HetNetsRepresentation]):
    """A representer that creates HetNets representations."""

    sampler: Sampler[In, Any, S]

    def __init__(
        self,
        predictor: HetNetsPredictor[In, Any],
        num_samples: int,
        sampling_strategy: SamplingStrategy = "sequential",
    ) -> None:
        """Initialize the HetNets representer.

        Args:
            predictor: The predictor to be used for sampling.
            num_samples: The number of samples to draw.
            sampling_strategy: How the samples should be computed.
        """
        super().__init__(predictor)
        self.sampler = Sampler(predictor, num_samples, sampling_strategy=sampling_strategy)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> HetNetsRepresentation:
        """Sample from the predictor for a given input."""
        sample = self.sampler.represent(*args, **kwargs)
        return create_het_nets_representation(sample)
