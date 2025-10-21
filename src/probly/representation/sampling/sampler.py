"""Model sampling and sample representer implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal, Unpack

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE
from probly.predictor import Predictor, predict
from probly.representation.representer import Representer
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, GlobalVariable, lazydispatch_traverser, traverse_with_state

from .sample import Sample, create_sample

type SamplingStrategy = Literal["sequential"]


sampling_preparation_traverser = lazydispatch_traverser[object](name="sampling_preparation_traverser")

CLEANUP_FUNCS = GlobalVariable[set[Callable[[], Any]]](name="CLEANUP_FUNCS")


@sampling_preparation_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch_sampler as torch_sampler  # noqa: PLC0414, PLC0415


@sampling_preparation_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax_sampler as flax_sampler  # noqa: PLC0414, PLC0415


def get_sampling_predictor[In, KwIn, Out](
    predictor: Predictor[In, KwIn, Out],
) -> tuple[Predictor[In, KwIn, Out], Callable[[], None]]:
    """Get the predictor to be used for sampling."""
    predictor, state = traverse_with_state(
        predictor,
        nn_compose(sampling_preparation_traverser),
        init={CLONE: False, CLEANUP_FUNCS: set()},
    )
    cleanup_funcs = state[CLEANUP_FUNCS]

    def cleanup() -> None:
        for func in cleanup_funcs:
            func()

    return predictor, cleanup


def sampler_factory[In, KwIn, Out](
    predictor: Predictor[In, KwIn, Out],
    num_samples: int = 1,
    strategy: SamplingStrategy = "sequential",
) -> Predictor[In, KwIn, list[Out]]:
    """Sample multiple predictions from the predictor."""

    def sampler(*args: In, **kwargs: Unpack[KwIn]) -> list[Out]:
        sampling_predictor, cleanup = get_sampling_predictor(predictor)
        try:
            if strategy == "sequential":
                return [predict(sampling_predictor, *args, **kwargs) for _ in range(num_samples)]
        finally:
            cleanup()

        msg = f"Unknown sampling strategy: {strategy}"
        raise ValueError(msg)

    return sampler


class Sampler[In, KwIn, Out](Representer[In, KwIn, Out]):
    """A representation predictor that creates representations from finite samples."""

    sampling_strategy: SamplingStrategy
    sample_factory: Callable[[Iterable[Out]], Sample[Out]]

    def __init__(
        self,
        predictor: Predictor[In, KwIn, Out],
        num_samples: int = 1,
        sampling_strategy: SamplingStrategy = "sequential",
        sample_factory: Callable[[Iterable[Out]], Sample[Out]] = create_sample,
    ) -> None:
        """Initialize the sampler.

        Args:
            predictor (Predictor[In, KwIn, Out]): The predictor to be used for sampling.
            num_samples (int, optional): The number of samples to draw.
            sampling_strategy (SamplingStrategy, optional): How the samples should be computed.
            sample_factory (Callable[[Iterable[Out]], Sample[Out]], optional): Factory to create the sample.
        """
        super().__init__(predictor)
        self.num_samples = num_samples
        self.sampling_strategy = sampling_strategy
        self.sample_factory = sample_factory

    def sample(self, *args: In, **kwargs: Unpack[KwIn]) -> Sample[Out]:
        """Sample from the predictor for a given input."""
        return self.sample_factory(
            sampler_factory(
                self.predictor,
                num_samples=self.num_samples,
                strategy=self.sampling_strategy,
            )(*args, **kwargs),
        )


class EnsembleSampler[In, KwIn, Out](Representer[In, KwIn, Iterable[Out]]):
    """A sampler that creates representations from ensemble predictions."""

    sample_factory: Callable[[Iterable[Out]], Sample[Out]]

    def __init__(
        self,
        predictor: Predictor[In, KwIn, Iterable[Out]],
        sample_factory: Callable[[Iterable[Out]], Sample[Out]] = create_sample,
    ) -> None:
        """Initialize the ensemble sampler.

        Args:
            predictor (Predictor[In, KwIn, Out]): The ensemble predictor.
            sample_factory (Callable[[Iterable[Out]], Sample[Out]], optional): Factory to create the sample.
        """
        super().__init__(predictor)
        self.sample_factory = sample_factory

    def sample(self, *args: In, **kwargs: Unpack[KwIn]) -> Sample[Out]:
        """Sample from the ensemble predictor for a given input."""
        return self.sample_factory(
            self.predictor(*args, **kwargs),
        )
