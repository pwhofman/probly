"""Model sampling and sample representer implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, override

from probly.lazy_types import FLAX_MODULE, SKLEARN_MODULE, TORCH_MODULE
from probly.predictor import EnsemblePredictor, Predictor, RandomPredictor, predict
from probly.representation.representer import Representer, representer
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, GlobalVariable, function_traverser, lazydispatch_traverser, traverse_with_state

from .sample import Sample, SampleFactory, create_sample

type SamplingStrategy = Literal["sequential"]


sampling_preparation_traverser = lazydispatch_traverser[object](name="sampling_preparation_traverser")

CLEANUP_FUNCS = GlobalVariable[set[Callable[[], Any]]](name="CLEANUP_FUNCS")


@sampling_preparation_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch_sampler as torch_sampler  # noqa: PLC0414, PLC0415


@sampling_preparation_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax_sampler as flax_sampler  # noqa: PLC0414, PLC0415


@sampling_preparation_traverser.delayed_register(SKLEARN_MODULE)
def _(_: type) -> None:
    from . import sklearn_sampler as sklearn_sampler  # noqa: PLC0414, PLC0415


def get_sampling_predictor[**In, Out](
    predictor: Predictor[In, Out],
) -> tuple[Predictor[In, Out], Callable[[], None]]:
    """Get the predictor to be used for sampling."""
    predictor, state = traverse_with_state(
        predictor,
        nn_compose(sampling_preparation_traverser, function_traverser),
        init={CLONE: False, CLEANUP_FUNCS: set()},
    )
    cleanup_funcs = state[CLEANUP_FUNCS]

    def cleanup() -> None:
        for func in cleanup_funcs:
            func()

    return predictor, cleanup


def sampler_factory[**In, Out](
    predictor: Predictor[In, Out],
    num_samples: int = 1,
    strategy: SamplingStrategy = "sequential",
) -> Predictor[In, list[Out]]:
    """Sample multiple predictions from the predictor."""

    def sampler(*args: In.args, **kwargs: In.kwargs) -> list[Out]:
        sampling_predictor, cleanup = get_sampling_predictor(predictor)
        try:
            if strategy == "sequential":
                return [predict(sampling_predictor, *args, **kwargs) for _ in range(num_samples)]
        finally:
            cleanup()

        msg = f"Unknown sampling strategy: {strategy}"
        raise ValueError(msg)

    return sampler


@representer.register(RandomPredictor)
class Sampler[**In, Out, S: Sample](Representer[Any, In, S]):
    """A representation predictor that creates representations from finite samples."""

    sampling_strategy: SamplingStrategy
    sample_factory: SampleFactory[Out, S]
    num_samples: int
    sample_axis: int

    def __init__(
        self,
        predictor: RandomPredictor[In, Out],
        num_samples: int,
        sampling_strategy: SamplingStrategy = "sequential",
        sample_factory: SampleFactory[Out, S] = create_sample,
        sample_axis: int = 1,
    ) -> None:
        """Initialize the sampler.

        Args:
            predictor: The predictor to be used for sampling.
            num_samples: The number of samples to draw.
            sampling_strategy: How the samples should be computed.
            sample_factory: Factory to create the sample.
            sample_axis: The axis along which samples are organized.
        """
        super().__init__(predictor)
        self.num_samples = num_samples
        self.sampling_strategy = sampling_strategy
        self.sample_factory = sample_factory
        self.sample_axis = sample_axis

    def predict(self, *args: In.args, **kwargs: In.kwargs) -> S:
        """Sample from the predictor for a given input."""
        return self.sample_factory(
            sampler_factory(
                self.predictor,
                num_samples=self.num_samples,
                strategy=self.sampling_strategy,
            )(*args, **kwargs),  # type: ignore[no-any-return]
            sample_axis=self.sample_axis,
        )

    @override
    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> S:
        return self.predict(*args, **kwargs)


@representer.register(EnsemblePredictor)
class EnsembleSampler[**In, Out, S: Sample](Representer[Any, In, S]):
    """A sampler that creates representations from ensemble predictions."""

    sample_factory: SampleFactory[Out, S]

    def __init__(
        self,
        predictor: EnsemblePredictor[In, Out],
        sample_factory: SampleFactory[Out, S] = create_sample,
        sample_axis: int = 1,
    ) -> None:
        """Initialize the ensemble sampler.

        Args:
            predictor: The ensemble predictor.
            sample_factory: Factory to create the sample.
            sample_axis: The axis along which samples are organized.
        """
        super().__init__(predictor)
        self.sample_factory = sample_factory
        self.sample_axis = sample_axis

    def predict(self, *args: In.args, **kwargs: In.kwargs) -> S:
        """Sample from the ensemble predictor for a given input."""
        return self.sample_factory(
            predict(self.predictor, *args, **kwargs),
            sample_axis=self.sample_axis,
        )

    @override
    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> S:
        return self.predict(*args, **kwargs)
