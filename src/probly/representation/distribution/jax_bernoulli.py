"""Jax-based Bernoulli distribution representation."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, override

import jax
from jax import numpy as jnp
from scipy.special import logit

from probly.representation.distribution._common import (
    BernoulliDistribution,
    BernoulliDistributionSample,
    create_bernoulli_distribution,
    create_bernoulli_distribution_from_logits,
)
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxLogitCategoricalDistribution,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.jax_functions import jax_stack
from probly.representation.sample.jax import JaxArraySample


class JaxBernoulliDistribution(BernoulliDistribution, JaxCategoricalDistribution, ABC): # ty:ignore[conflicting-metaclass]
    """A Bernoulli distribution represented as a categorical dstribution with 2 classes."""


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxProbabilityBernoulliDistribution(JaxProbabilityCategoricalDistribution, JaxBernoulliDistribution):
    """A Bernoulli distribution represented by the probability of class 1."""

    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}

    def __post_init__(self) -> None:
        """Validate probability parameters."""
        if not isinstance(self.array, jax.Array):
            msg = "probabilities must be a jax Array."
            raise TypeError(msg)
        if jnp.any((self.array < 0.0) | (self.array > 1.0)):
            msg = "Bernoulli probabilities must be in [0, 1]."
            raise ValueError(msg)

    @override
    @property
    def unnormalized_probabilities(self) -> jax.Array:
        return jax_stack((1.0 - self.array, self.array), axis=-1)

    @override
    @property
    def logits(self) -> jax.Array:
        positive = logit(self.array)
        return jax_stack((jnp.zeros_like(positive), positive), axis=-1)

    @override
    def to_categorical(self) -> JaxProbabilityCategoricalDistribution:
        return JaxProbabilityCategoricalDistribution(self.probabilities)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxLogitBernoulliDistribution(JaxLogitCategoricalDistribution, JaxBernoulliDistribution):
    """A Bernoulli distribution represented by class-1 log-odds."""

    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}

    def __post_init__(self) -> None:
        """Validate logit parameters."""
        if not isinstance(self.array, jax.Array):
            msg = "logits must be a jax Array."
            raise TypeError(msg)

    @override
    @property
    def logits(self) -> jax.Array:
        return jax_stack((jnp.zeros_like(self.array), self.array), axis=-1)

    @override
    def to_categorical(self) -> JaxLogitCategoricalDistribution:
        return JaxLogitCategoricalDistribution(self.logits)


class JaxBernoulliDistributionSample( # ty:ignore[conflicting-metaclass]
    BernoulliDistributionSample[JaxBernoulliDistribution],
    JaxArraySample[JaxBernoulliDistribution],
):
    """Sample type for jax Bernoulli distributins."""

    sample_space: ClassVar[type[BernoulliDistribution]] = JaxBernoulliDistribution


@create_bernoulli_distribution.register((list, tuple))
def _create_jax_bernoulli_distribution_from_sequence(data: list[Any] | tuple[Any, ...]) -> BernoulliDistribution:
    return _create_jax_bernoulli_distribution(jnp.asarray(data))


@create_bernoulli_distribution.register(jax.Array)
def _create_jax_bernoulli_distribution(data: jax.Array) -> BernoulliDistribution:
    if data.ndim >= 2 and data.shape[-1] <= 2:
        data = data[..., -1]
    return JaxProbabilityBernoulliDistribution(data)


@create_bernoulli_distribution_from_logits.register(jax.Array)
def _create_jax_bernoulli_distribution_from_logits(data: jax.Array) -> BernoulliDistribution:
    if data.ndim >= 2 and data.shape[-1] == 2:
        data = data[..., -1] - data[..., 0]
    elif data.ndim >= 2 and data.shape[-1] == 1:
        data = data[..., -1]
    return JaxLogitBernoulliDistribution(data)