"""Model sampling and sample representer implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Unpack

from probly.lazy_types import TORCH_MODULE
from probly.predictor import Predictor, predict
from probly.representation.representer import Representer
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, GlobalVariable, lazy_singledispatch_traverser, traverse_with_state

from .sample import Sample, create_sample

type SamplingStrategy = Literal["sequential"]


sampling_preparation_traverser = lazy_singledispatch_traverser[object](name="sampling_preparation_traverser")

CLEANUP_FUNCS = GlobalVariable[set[Callable[[], Any]]](name="CLEANUP_FUNCS")


@sampling_preparation_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch_sampler as torch_sampler  # noqa: PLC0414, PLC0415


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


def sample_predictions[In, KwIn, Out](
    predictor: Predictor[In, KwIn, Out],
    *args: In,
    num_samples: int = 1,
    strategy: SamplingStrategy = "sequential",
    **kwargs: Unpack[KwIn],
) -> list[Out]:
    """Sample multiple predictions from the predictor."""
    predictor, cleanup = get_sampling_predictor(predictor)
    try:
        if strategy == "sequential":
            return [predict(predictor, *args, **kwargs) for _ in range(num_samples)]
    finally:
        cleanup()

    msg = f"Unknown sampling strategy: {strategy}"
    raise ValueError(msg)


class Sampler[In, KwIn, Out](Representer[In, KwIn, Out]):
    """A representation predictor that creates representations from finite samples."""

    sampling_strategy: SamplingStrategy
    sampler_factory: Callable[[list[Out]], Sample[Out]]

    def __init__(
        self,
        predictor: Predictor[In, KwIn, Out],
        num_samples: int = 1,
        sampling_strategy: SamplingStrategy = "sequential",
        sampler_factory: Callable[[list[Out]], Sample[Out]] = create_sample,
    ) -> None:
        """Initialize the sampler.

        Args:
            predictor (Predictor[In, KwIn, Out]): The predictor to be used for sampling.
            sampling_strategy (SamplingStrategy, optional): How the samples should be computed.
            sampler_factory (Callable[[list[Out]], Sample[Out]], optional): Factory to create the sample representation.
        """
        super().__init__(predictor)
        self.num_samples = num_samples
        self.sampling_strategy = sampling_strategy
        self.sampler_factory = sampler_factory

    def sample(self, *args: In, **kwargs: Unpack[KwIn]) -> Sample[Out]:
        """Sample from the predictor for a given input."""
        return self.sampler_factory(
            sample_predictions(
                self.predictor,
                *args,
                num_samples=self.num_samples,
                strategy=self.sampling_strategy,
                **kwargs,
            ),
        )
