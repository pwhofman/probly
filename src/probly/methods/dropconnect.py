"""DropConnect method for uncertainty quantification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

from probly.representation.sampling.sample import Sample
from probly.representation.sampling.sampler import Sampler
from probly.transformation.dropconnect.common import dropconnect

if TYPE_CHECKING:
    from probly.predictor import Predictor


class DropConnect[In, KwIn, Out, S: Sample]:
    """DropConnect for uncertainty quantification.

    Based on :cite:t:`mobiny2021dropconnect`.
    """

    def __init__(
        self,
        base: Predictor[In, KwIn, Out],
        p: float = 0.25,
        num_samples: int = 100,
    ) -> None:
        """Initialize the DropConnect method.

        Args:
            base: The base model
            p: The dropout probability
            num_samples: The number of samples to draw
        """
        self._model = dropconnect(base, p=p)
        self._sampler: Sampler[In, KwIn, Out, S] = Sampler(
            predictor=self._model,
            num_samples=num_samples,
        )
        self._p = p
        self._num_samples = num_samples

    def predict(self, *args: In, **kwargs: Unpack[KwIn]) -> S:
        """Run stochastic forward passes and return an uncertainty representation."""
        return self._sampler.predict(*args, **kwargs)

    @property
    def model(self) -> Predictor[In, KwIn, Out]:
        """Get the base model."""
        return self._model

    @property
    def p(self) -> float:
        """Get the dropout probability."""
        return self._p

    @property
    def num_samples(self) -> int:
        """Get the number of samples."""
        return self._num_samples
