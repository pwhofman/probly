"""Implementation of a Bayesian neural network."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representation.sampling.common_sample import Sample
from probly.representation.sampling.sampler import Sampler
from probly.transformation import bayesian


class BayesianNeuralNetwork[**In, Out]:
    """Bayesian neural network implementation based on :cite:`blundellWeightUncertainty2015`."""

    def __init__(
        self,
        base: Predictor[In, Out],
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
        self.num_samples = num_samples

    def sampler(self, num_samples: int | None = None) -> Sampler[In, Out, Sample[Out]]:
        """Create a sampler for the Bayesian neural network."""
        if num_samples is None:
            num_samples = self.num_samples
        return Sampler(self.model, num_samples=num_samples)

    def predict(self, *args: In.args, **kwargs: In.kwargs) -> Sample[Out]:
        """Predict the output for a given input."""
        return self.sampler().predict(*args, **kwargs)
