"""Sampling preparation for torch."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

from laplace.baselaplace import BaseLaplace
import torch.nn

from probly.layers.torch import DropConnectLinear
from probly.representation.distribution import (
    CategoricalDistribution,
    create_categorical_distribution,
)
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.sample import Sample, SampleFactory, create_sample
from probly.representer._representer import representer

from ._common import CLEANUP_FUNCS, Sampler, SamplingStrategy, sampling_preparation_traverser

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from flextype.isinstance import LazyType
    from pytraverse import State


def _enforce_train_mode(obj: torch.nn.Module, state: State) -> tuple[torch.nn.Module, State]:
    if not obj.training:
        obj.train()
        state[CLEANUP_FUNCS].add(lambda: obj.train(False))
        return obj, state
    return obj, state


def register_forced_train_mode(cls: LazyType) -> None:
    """Register a class to be forced into train mode during sampling.

    This enables Monte Carlo sampling techniques like MC Dropout :cite:`galDropoutBayesian2016` or DropConnect :cite:
    `mobinyDropConnectEffective2019`.
    """
    sampling_preparation_traverser.register(
        cls,
        _enforce_train_mode,
    )


register_forced_train_mode(
    torch.nn.Dropout
    | torch.nn.Dropout1d
    | torch.nn.Dropout2d
    | torch.nn.Dropout3d
    | torch.nn.AlphaDropout
    | torch.nn.FeatureAlphaDropout
    | DropConnectLinear,
)


@representer.register(BaseLaplace)
class LaplaceSampler[**In, S: Sample](Sampler[In, CategoricalDistribution, S]):
    """Sampler over ``BaseLaplace.predictive_samples`` (classification only; see laplace-torch docs)."""

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
        """``pred_type`` is forwarded to ``BaseLaplace.__call__``; rest inherited from :class:`Sampler`."""
        super().__init__(predictor, num_samples, sampling_strategy, sample_factory, sample_axis)
        self.pred_type = pred_type

    def _bulk_predictive_samples(self, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
        """Single bulk call to ``BaseLaplace.predictive_samples`` for the configured ``pred_type``."""
        predictor = cast("Any", self.predictor)
        if predictor.likelihood != "classification":
            msg = f"only likelihood='classification' is supported, got {predictor.likelihood!r}"
            raise NotImplementedError(msg)
        if not args:
            msg = "represent expects the input batch as the first argument"
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
        """Bulk-sample from the posterior; returns a :class:`TorchCategoricalDistributionSample`."""
        samples_tensor = self._bulk_predictive_samples(*args, **kwargs)
        # samples_tensor shape: (num_samples, batch, num_classes); class axis stays at -1.
        target_dim = self.sample_axis if self.sample_axis >= 0 else samples_tensor.ndim - 1 + self.sample_axis
        cat = create_categorical_distribution(torch.moveaxis(samples_tensor, 0, target_dim))
        return TorchCategoricalDistributionSample(tensor=cat, sample_dim=target_dim)  # ty:ignore[invalid-argument-type, invalid-return-type]
