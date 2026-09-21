"""NumPy-based Bernoulli distribution representation."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, ClassVar, override

import numpy as np
from scipy.special import logit

from probly.representation.distribution._common import (
    BernoulliDistribution,
    BernoulliDistributionSample,
    create_bernoulli_distribution,
    create_bernoulli_distribution_from_logits,
)
from probly.representation.distribution.numpy_categorical import (
    NumpyCategoricalDistribution,
    NumpyLogitCategoricalDistribution,
    NumpyProbabilityCategoricalDistribution,
)
from probly.representation.sample.numpy import NumpySample


class NumpyBernoulliDistribution(BernoulliDistribution, NumpyCategoricalDistribution, ABC):  # ty:ignore[conflicting-metaclass]
    """A Bernoulli distribution represented as a categorical distribution with 2 classes."""


@dataclass(frozen=True, slots=True, weakref_slot=True)
class NumpyProbabilityBernoulliDistribution(NumpyProbabilityCategoricalDistribution, NumpyBernoulliDistribution):
    """A Bernoulli distribution represented by the probability of class 1."""

    array: np.ndarray = field()
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}

    def __post_init__(self) -> None:
        """Validate probability parameters."""
        if not isinstance(self.array, np.ndarray):
            msg = "probabilities must be a numpy ndarray."
            raise TypeError(msg)
        if np.any((self.array < 0.0) | (self.array > 1.0)):
            msg = "Bernoulli probabilities must be in [0, 1]."
            raise ValueError(msg)

    @override
    @property
    def unnormalized_probabilities(self) -> np.ndarray:
        return np.stack((1.0 - self.array, self.array), axis=-1)

    @override
    @property
    def logits(self) -> np.ndarray:
        positive = logit(self.array)
        return np.stack((np.zeros_like(positive), positive), axis=-1)

    @override
    def to_categorical(self) -> NumpyProbabilityCategoricalDistribution:
        return NumpyProbabilityCategoricalDistribution(self.probabilities)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class NumpyLogitBernoulliDistribution(NumpyLogitCategoricalDistribution, NumpyBernoulliDistribution):
    """A Bernoulli distribution represented by class-1 log-odds."""

    array: np.ndarray = field()
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}

    def __post_init__(self) -> None:
        """Validate logit parameters."""
        if not isinstance(self.array, np.ndarray):
            msg = "logits must be a numpy ndarray."
            raise TypeError(msg)

    @override
    @property
    def logits(self) -> np.ndarray:
        return np.stack((np.zeros_like(self.array), self.array), axis=-1)

    @override
    def to_categorical(self) -> NumpyLogitCategoricalDistribution:
        return NumpyLogitCategoricalDistribution(self.logits)


class NumpyBernoulliDistributionSample(  # ty:ignore[conflicting-metaclass]
    BernoulliDistributionSample[NumpyBernoulliDistribution],
    NumpySample[NumpyBernoulliDistribution],
):
    """Sample type for array Bernoulli distributions."""

    sample_space: ClassVar[type[BernoulliDistribution]] = NumpyBernoulliDistribution


@create_bernoulli_distribution.register((list, tuple))
def _numpy_create_bernoulli_distribution_from_sequence(data: list[Any] | tuple[Any, ...]) -> BernoulliDistribution:
    return _numpy_create_bernoulli_distribution(np.asarray(data))


@create_bernoulli_distribution.register(np.ndarray)
def _numpy_create_bernoulli_distribution(data: np.ndarray) -> BernoulliDistribution:
    if data.ndim >= 2 and data.shape[-1] <= 2:
        data = data[..., -1]
    return NumpyProbabilityBernoulliDistribution(data)


@create_bernoulli_distribution_from_logits.register(np.ndarray)
def _numpy_create_bernoulli_distribution_from_logits(data: np.ndarray) -> BernoulliDistribution:
    if data.ndim >= 2 and data.shape[-1] == 2:
        data = data[..., -1] - data[..., 0]
    elif data.ndim >= 2 and data.shape[-1] == 1:
        data = data[..., -1]
    return NumpyLogitBernoulliDistribution(data)
