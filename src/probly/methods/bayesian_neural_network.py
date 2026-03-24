"""Implementation of a Bayesian neural network."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representation.sampling.common_sample import Sample
from probly.representation.sampling.sampler import Sampler
from probly.transformation import bayesian


class BayesianNeuralNetwork[In, KwIn, Out]:
    """Bayesian neural network implementation based on :cite:`blundellWeightUncertainty2015`."""

    def __init__(
        self,
        base: Predictor[In, KwIn, Out],
        num_samples: int,
        use_base_weights: bool,
        posterior_std: float,
        prior_mean: float,
        prior_std: float,
    ) -> None:
        """Initialize the Bayesian neural network with a base predictor and the number of members in the ensemble."""
        self.model = bayesian(
            base,
            use_base_weights=use_base_weights,
            posterior_std=posterior_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
        )
        self._sampler = Sampler(self.model, num_samples=num_samples)

    def predict(self, *args: In, **kwargs: Unpack[KwIn]) -> Sample[Out]:
        """Predict the output for a given input."""
        return self._sampler.predict(*args, **kwargs)
