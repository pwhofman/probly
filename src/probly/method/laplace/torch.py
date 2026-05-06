"""Laplace approximation representer for the laplace-torch package."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

from laplace.baselaplace import BaseLaplace
import torch

from probly.representation.distribution import CategoricalDistribution, create_categorical_distribution
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.sample import Sample, SampleFactory, create_sample
from probly.representer._representer import representer
from probly.representer.sampler._common import Sampler, SamplingStrategy

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any


@representer.register(BaseLaplace)
class LaplaceRepresenter[**In, S: Sample](Sampler[In, CategoricalDistribution, S]):
    """Representer over ``BaseLaplace.predictive_samples`` (classification only; see laplace-torch docs).

    Args:
        predictor: A fitted ``BaseLaplace`` instance.
        num_samples: Number of posterior samples to draw.
        pred_type: Forwarded to ``BaseLaplace.predictive_samples``; typically ``"glm"`` or ``"nn"``.
        sampling_strategy: How repeated predictions are computed.
        sample_factory: Factory used to build the returned ``Sample``.
        sample_axis: Axis along which samples are stacked in the output tensor.
    """

    pred_type: str

    def __init__(
        self,
        predictor: BaseLaplace,
        num_samples: int,
        pred_type: str = "glm",
        sampling_strategy: SamplingStrategy = "sequential",
        sample_factory: SampleFactory[CategoricalDistribution, S] = create_sample,  # ty:ignore[invalid-parameter-default]
        sample_axis: int = -1,
    ) -> None:
        """Initialize the Laplace representer."""
        super().__init__(predictor, num_samples, sampling_strategy, sample_factory, sample_axis)
        self.pred_type = pred_type

    def _bulk_predictive_samples(self, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
        """Single bulk call to ``BaseLaplace.predictive_samples``."""
        predictor = cast("Any", self.predictor)
        if predictor.likelihood != "classification":
            msg = f"only likelihood='classification' is supported, got {predictor.likelihood!r}"
            raise NotImplementedError(msg)
        if not args:
            msg = "represent expects the input batch as the first positional argument"
            raise TypeError(msg)
        passthrough = {k: v for k, v in kwargs.items() if k != "n_samples"}
        return predictor.predictive_samples(
            args[0], pred_type=self.pred_type, n_samples=self.num_samples, **passthrough
        )

    @override
    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Iterable[CategoricalDistribution]:
        samples_tensor = self._bulk_predictive_samples(*args, **kwargs)
        return [create_categorical_distribution(samples_tensor[i]) for i in range(self.num_samples)]

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> S:
        """Bulk-sample from the posterior and return a :class:`TorchCategoricalDistributionSample`."""
        samples_tensor = self._bulk_predictive_samples(*args, **kwargs)
        # samples_tensor shape: (num_samples, batch, num_classes); move sample axis to target_dim.
        target_dim = self.sample_axis if self.sample_axis >= 0 else samples_tensor.ndim - 1 + self.sample_axis
        cat = create_categorical_distribution(torch.moveaxis(samples_tensor, 0, target_dim))
        return TorchCategoricalDistributionSample(  # ty:ignore[invalid-return-type]
            tensor=cat,  # ty:ignore[invalid-argument-type]
            sample_dim=target_dim,
        )
