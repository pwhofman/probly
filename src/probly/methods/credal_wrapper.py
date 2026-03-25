"""Implementation of the Credal Wrapper method."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.representation.credal_set import create_probability_intervals

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representation.credal_set.common import ProbabilityIntervalsCredalSet
from probly.representation.sampling.sample import create_sample
from probly.representation.sampling.sampler import EnsembleSampler
from probly.transformation.ensemble.common import ensemble


class CredalWrapper[**In, Out]:
    """Credal wrapper method."""

    def __init__(self, base: Predictor[In, Out], num_members: int) -> None:
        """Initialize the credal wrapper method.

        Args:
            base: The base predictor to be used for the ensemble.
            num_members: The number of members in the ensemble.
        """
        self.ensemble = ensemble(base, num_members=num_members)
        self._sampler = EnsembleSampler(self.ensemble, sample_factory=create_sample)

    def predict(self, *args: In.args, **kwargs: In.kwargs) -> ProbabilityIntervalsCredalSet[Out]:
        """Predict a credal set for the given input."""
        sample = self._sampler.predict(*args, **kwargs)
        cset = create_probability_intervals(sample)
        return cset
