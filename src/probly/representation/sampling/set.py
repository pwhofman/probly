"""Model set representer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from probly.predictor import Predictor
from probly.representation.representer import Representer
from probly.representation.sampling.credal_set import CredalSet, create_credal_set
from probly.representation.sampling.sampler import SamplingStrategy, sampler_factory


class Set[In, KwIn, Out](Representer[In, KwIn, Out]):
    """A representation predictor that creates credal sets from finite samples."""

    sampling_strategy: SamplingStrategy
    sample_factory: Callable[[Iterable[Out]], CredalSet[Out]]

    def __init__(
        self,
        predictor: Predictor[In, KwIn, Out],
        sampling_strategy: SamplingStrategy = "sequential",
        sample_factory: Callable[[Iterable[Out]], CredalSet[Out]] = create_credal_set,
    ) -> None:
        """Initialize the sampler.

        Args:
            predictor (Predictor[In, KwIn, Out]): The predictor to be used for sampling.
            sampling_strategy (SamplingStrategy, optional): How the samples should be computed.
            sample_factory (Callable[[Iterable[Out]], Sample[Out]], optional): Factory to create the sample.
        """
        super().__init__(predictor)
        self.sampling_strategy = sampling_strategy
        self.sample_factory = sample_factory

    def predict(self, *args: In, num_samples: int, **kwargs: Unpack[KwIn]) -> CredalSet[Out]:
        """Sample from the predictor for a given input to create a credal set."""
        return self.sample_factory(
            sampler_factory(
                self.predictor,
                num_samples=num_samples,
                strategy=self.sampling_strategy,
            )(*args, **kwargs),
        )
