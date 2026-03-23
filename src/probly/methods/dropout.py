"""Monte Carlo Dropout method for uncertainty quantification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

from probly.representation.sampling.sample import Sample
from probly.representation.sampling.sampler import Sampler
from probly.transformation.dropout.common import dropout

if TYPE_CHECKING:
    from probly.predictor import Predictor


class Dropout[In, KwIn, Out, S: Sample]:
    """Dropout for uncertainty quantification.

    Based on :cite:t:`galDropoutBayesian2016`.
    """

    model: Predictor[In, KwIn, Out]

    def __init__(
        self,
        base: Predictor[In, KwIn, Out],
        p: float = 0.25,
        num_samples: int = 100,
    ) -> None:
        """Initialize the MCDropout method.

        Args:
            base: The base model
            p: The dropout probability
            num_samples: The number of samples to draw
        """
        self.model = dropout(base, p=p)
        self._sampler: Sampler[In, KwIn, Out, S] = Sampler(
            predictor=self.model,
            num_samples=num_samples,
        )
        self.p = p
        self.num_samples = num_samples

    def predict(self, *args: In, **kwargs: Unpack[KwIn]) -> S:
        """Run stochastic forward passes and return an uncertainty representation."""
        return self._sampler.predict(*args, **kwargs)
