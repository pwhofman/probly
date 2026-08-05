"""Jax-based distribution representation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, override

import jax
import jax.numpy as jnp

from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.distribution._common import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    create_categorical_distribution,
    create_categorical_distribution_from_logits,
)
from probly.representation.jax_functions import jax_average, jax_mean
from probly.representation.sample.jax import JaxArraySample

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax._src.typing import ArrayLike


class JaxCategoricalDistribution(CategoricalDistribution, JaxAxisProtected[jax.Array], ABC):
    """A categorical distribution stored as a jax array.

    Shape: (..., num_classes)
    The last axis represents the category dimensions.
    """

    @property
    @abstractmethod
    def array(self) -> jax.Array:
        """Get the underlying array representing the categorical distribution."""

    @override
    def _postprocess_protected_values(self, values: dict[str, jax.Array], func: Callable) -> dict[str, jax.Array]:
        if func in (jax_mean, jax_average):
            values["array"] = self.probabilities

        return values

    @override
    def with_protected_values(
        self, values: dict[str, Any], func: Callable | None = None
    ) -> JaxAxisProtected[jax.Array]:
        """Return a copy with a replaced primary protected value."""
        if func in (jax_mean, jax_average) and not isinstance(self, JaxProbabilityCategoricalDistribution):
            return JaxProbabilityCategoricalDistribution(array=values["array"])

        return super().with_protected_values(values, func)

    @override
    @property
    def unnormalized_probabilities(self) -> jax.Array:
        logits = self.logits
        return jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))

    @override
    @property
    def probabilities(self) -> jax.Array:
        unnormalized_probabilities = self.unnormalized_probabilities
        sums = jnp.sum(unnormalized_probabilities, axis=-1, keepdims=True)
        return unnormalized_probabilities / sums

    @override
    @property
    def logits(self) -> jax.Array:
        return jnp.log(self.unnormalized_probabilities)

    @override
    @property
    def log_probabilities(self) -> jax.Array:
        return jax.nn.log_softmax(self.logits, axis=-1)

    @override
    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return self.unnormalized_probabilities.shape[-1]

    @override
    def sample(
        self,
        num_samples: int = 1,
        prng_key: ArrayLike | None = None,
    ) -> JaxArraySample[jax.Array]:
        """Sample from the categorical distribution (Jax backend)."""
        if prng_key is None:
            prng_key = jax.random.key(0)

        samples = jax.random.categorical(prng_key, self.logits, axis=-1, shape=(num_samples, *self.shape))
        return JaxArraySample(array=samples, sample_axis=0)


@create_categorical_distribution.register(jax.Array)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxProbabilityCategoricalDistribution(JaxCategoricalDistribution):
    """A categorical distribution represented by unnormalized probabilities."""

    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Callable]] = {jax_mean, jax_average}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.array, jax.Array):
            msg = "probabilities must be a jax Array."
            raise TypeError(msg)

        if self.array.ndim < 1:
            msg = "probabilities must have at least one dimension."
            raise ValueError(msg)
        if jnp.any(self.array < 0):
            msg = "Relative probabilities must be a non-negative."
            raise ValueError(msg)

    @override
    @property
    def unnormalized_probabilities(self) -> jax.Array:
        return self.array

    @override
    def __eq__(self, value: Any) -> jax.Array:  # ty: ignore[invalid-method-override] # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, JaxCategoricalDistribution):
            eq = jnp.equal(self.probabilities, value.probabilities)
        else:
            eq = jnp.equal(self.array, value)
        return jnp.all(eq, axis=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


@create_categorical_distribution_from_logits.register(jax.Array)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxLogitCategoricalDistribution(JaxCategoricalDistribution):
    """A categorical distribution represented by logits."""

    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Callable]] = {jax_mean, jax_average}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.array, jax.Array):
            msg = "logits must be a jax array."
            raise TypeError(msg)

        if self.array.ndim < 1:
            msg = "logits must have at least one dimension."
            raise ValueError(msg)

    @override
    @property
    def logits(self) -> jax.Array:
        return self.array

    @override
    def __eq__(self, value: Any) -> jax.Array:  # ty: ignore[invalid-method-override] # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, JaxCategoricalDistribution):
            eq = jnp.equal(self.log_probabilities, value.log_probabilities)
        else:
            eq = jnp.equal(self.array, value)
        return jnp.all(eq, axis=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` binding at runtime. ``object``'s
        hash gives per-instance indentity semantics.
        """
        return object.__hash__(self)


class JaxCategoricalDistributionSample(  # ty:ignore[conflicting-metaclass]
    CategoricalDistributionSample[JaxCategoricalDistribution],
    JaxArraySample[JaxCategoricalDistribution],
):
    """Sample type for empirical second-order categorical distributions."""

    sample_space: ClassVar[type[CategoricalDistribution]] = JaxCategoricalDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)
