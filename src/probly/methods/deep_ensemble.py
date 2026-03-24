"""Implementation of the Deep Ensemble method."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representation.sampling.sample import Sample

from probly.representation.sampling.common_sample import create_sample
from probly.representation.sampling.sampler import EnsembleSampler
from probly.transformation import ensemble


class DeepEnsemble[**In, Out]:
    """Deep ensemble class."""

    def __init__(self, base: Predictor[In, Out], num_members: int) -> None:
        """Initialize the deep ensemble method."""
        self.ensemble = ensemble(base, num_members=num_members)
        self._sampler = EnsembleSampler(self.ensemble, sample_factory=create_sample)

    def predict(self, *args: In.args, **kwargs: In.kwargs) -> Sample[Out]:
        """Predict a sample for the given input."""
        return self._sampler.sample(*args, **kwargs)
