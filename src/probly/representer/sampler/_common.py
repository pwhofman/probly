"""Model sampling and sample representer implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal, override

from probly.predictor import IterablePredictor, Predictor, RandomPredictor, predict
from probly.predictor._common import RandomRepresentationPredictor
from probly.representation.sample import Sample, SampleFactory, create_sample
from probly.representer._representer import Representer, representer
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, GlobalVariable, flexdispatch_traverser, function_traverser, traverse_with_state

type SamplingStrategy = Literal["sequential"]


sampling_preparation_traverser = flexdispatch_traverser[object](name="sampling_preparation_traverser")

CLEANUP_FUNCS = GlobalVariable[set[Callable[[], Any]]](name="CLEANUP_FUNCS")


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


@IterablePredictor.register_factory
def sampler_factory[**In, Out](
    predictor: Predictor[In, Out],
    num_samples: int = 1,
    strategy: SamplingStrategy = "sequential",
) -> Callable[In, list[Out]]:
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


@representer.register(IterablePredictor)
class IterableSampler[**In, Out, S: Sample](Representer[Any, In, Out, S]):
    """A sampler that creates representations from ensemble predictions."""

    sample_factory: SampleFactory[Out, S]
    sample_axis: int

    def __init__(
        self,
        predictor: IterablePredictor[In, Out],
        sample_factory: SampleFactory[Out, S] = create_sample,  # ty:ignore[invalid-parameter-default]
        sample_axis: int = -1,
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

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Iterable[Out]:
        """Predict multiple outputs from the ensemble predictor."""
        return predict(self.predictor, *args, **kwargs)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> S:
        """Sample from the ensemble predictor for a given input."""
        return self.sample_factory(
            self._predict(*args, **kwargs),
            sample_axis=self.sample_axis,
        )


@representer.register(RandomPredictor | RandomRepresentationPredictor)
class Sampler[**In, Out, S: Sample](IterableSampler[In, Out, S]):
    """A representation predictor that creates representations from finite samples."""

    sampling_strategy: SamplingStrategy
    num_samples: int

    def __init__(
        self,
        predictor: RandomPredictor[In, Out],
        num_samples: int,
        sampling_strategy: SamplingStrategy = "sequential",
        sample_factory: SampleFactory[Out, S] = create_sample,  # ty:ignore[invalid-parameter-default]
        sample_axis: int = -1,
    ) -> None:
        """Initialize the sampler.

        Args:
            predictor: The predictor to be used for sampling.
            num_samples: The number of samples to draw.
            sampling_strategy: How the samples should be computed.
            sample_factory: Factory to create the sample.
            sample_axis: The axis along which samples are organized.
        """
        super().__init__(predictor, sample_factory, sample_axis)
        self.num_samples = num_samples
        self.sampling_strategy = sampling_strategy

    @override
    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Iterable[Out]:
        """Sample from the predictor for a given input."""
        return sampler_factory(
            self.predictor,
            num_samples=self.num_samples,
            strategy=self.sampling_strategy,
        )(*args, **kwargs)
