"""Torch-based Bernoulli distribution representation."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, override

import torch

from probly.representation.distribution._common import (
    BernoulliDistribution,
    BernoulliDistributionSample,
    create_bernoulli_distribution,
    create_bernoulli_distribution_from_logits,
)
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchLogitCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.sample.torch import TorchSample


class TorchBernoulliDistribution(BernoulliDistribution, TorchCategoricalDistribution, ABC):  # ty:ignore[conflicting-metaclass]
    """A Bernoulli distribution represented as a categorical distribution with 2 classes."""


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchProbabilityBernoulliDistribution(TorchProbabilityCategoricalDistribution, TorchBernoulliDistribution):
    """A Bernoulli distribution represented by the probability of class 1."""

    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 0}

    def __post_init__(self) -> None:
        """Validate probability parameters."""
        if not isinstance(self.tensor, torch.Tensor):
            msg = "probabilities must be a torch tensor."
            raise TypeError(msg)
        if torch.any((self.tensor < 0.0) | (self.tensor > 1.0)):
            msg = "Bernoulli probabilities must be in [0, 1]."
            raise ValueError(msg)

    @override
    @property
    def unnormalized_probabilities(self) -> torch.Tensor:
        return torch.stack((1.0 - self.tensor, self.tensor), dim=-1)

    @override
    @property
    def logits(self) -> torch.Tensor:
        positive = torch.logit(self.tensor)
        return torch.stack((torch.zeros_like(positive), positive), dim=-1)

    @override
    def to_categorical(self) -> TorchCategoricalDistribution:
        return TorchProbabilityCategoricalDistribution(torch.as_tensor(self.probabilities))


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchLogitBernoulliDistribution(TorchLogitCategoricalDistribution, TorchBernoulliDistribution):
    """A Bernoulli distribution represented by class-1 log-odds."""

    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 0}

    def __post_init__(self) -> None:
        """Validate logit parameters."""
        if not isinstance(self.tensor, torch.Tensor):
            msg = "logits must be a torch tensor."
            raise TypeError(msg)

    @override
    @property
    def logits(self) -> torch.Tensor:
        return torch.stack((torch.zeros_like(self.tensor), self.tensor), dim=-1)

    @override
    def to_categorical(self) -> TorchCategoricalDistribution:
        return TorchLogitCategoricalDistribution(torch.as_tensor(self.logits))


class TorchBernoulliDistributionSample(  # ty:ignore[conflicting-metaclass]
    BernoulliDistributionSample[TorchBernoulliDistribution],
    TorchSample[TorchBernoulliDistribution],
):
    """Sample type for torch Bernoulli distributions."""

    sample_space: ClassVar[type[BernoulliDistribution]] = TorchBernoulliDistribution


@create_bernoulli_distribution.register(torch.Tensor)
def _create_torch_bernoulli_distribution(data: torch.Tensor) -> BernoulliDistribution:
    if data.ndim >= 2 and data.shape[-1] <= 2:
        data = data[..., -1]
    return TorchProbabilityBernoulliDistribution(data)


@create_bernoulli_distribution_from_logits.register(torch.Tensor)
def _create_torch_bernoulli_distribution_from_logits(data: torch.Tensor) -> BernoulliDistribution:
    if data.ndim >= 2 and data.shape[-1] == 2:
        data = data[..., -1] - data[..., 0]
    elif data.ndim >= 2 and data.shape[-1] == 1:
        data = data[..., -1]
    return TorchLogitBernoulliDistribution(data)
