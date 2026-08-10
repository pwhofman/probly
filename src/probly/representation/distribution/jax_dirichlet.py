"""Jax-based distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

import jax
from jax import numpy as jnp

from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.distribution._common import DirichletDistribution
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.jax_functions import jax_average, jax_mean, jax_sum
from probly.representation.sample.jax import JaxArraySample

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import DTypeLike
    from jax._src.typing import ArrayLike


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxDirichletDistribution(
    JaxAxisProtected[jax.Array],
    DirichletDistribution[JaxCategoricalDistribution],
):
    """A Dirichlet distribution stored as a jax Array.
    
    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    alphas: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"alphas": 1}
    permitted_functions: ClassVar[set[Callable]] = {jax_mean, jax_sum, jax_average}
    permitted_ufuncs: ClassVar[dict[jnp.ufunc, list[str]]] = {
        jnp.add: ["__call__"],
        jnp.subtract: ["__call__"],
    }

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.alphas, jax.Array):
            msg = "alphas must be a jax Array."
            raise TypeError(msg)

        if self.alphas.ndim < 1:
            msg = "alphas must have at least one dimension."
            raise ValueError(msg)

        if jnp.any(self.alphas <= 0):
            msg = "alphas must be strictly positive."
            raise ValueError(msg)

        if self.alphas.shape[-1] < 2:
            msg = "Dirichlet distribution requires at least 2 classes."
            raise ValueError(msg)

    @classmethod
    def from_array(cls, alphas: jax.Array | list, dtype: DTypeLike | None = None) -> Self:
        """Create a Dirichlet distribution from an array or list."""
        return cls(alphas=jnp.asarray(alphas, dtype=dtype))

    @property
    def mean(self) -> JaxCategoricalDistribution:
        """Return the mean of the Dirichlet distribution."""
        return JaxProbabilityCategoricalDistribution(self.alphas)

    @override
    def sample(
        self,
        num_samples: int = 1,
        prng_key: ArrayLike | None = None,
    ) -> JaxArraySample[JaxCategoricalDistribution]:
        """Sample from the Dirichlet distribution (Jax backend)."""
        if prng_key is None:
            prng_key = jax.random.key(0)

        gammas = jax.random.gamma(
            prng_key,
            self.alphas,
            shape = (num_samples, *self.alphas.shape),
        )
        return JaxArraySample(array=JaxProbabilityCategoricalDistribution(gammas), sample_axis=0)

    @override
    def _postprocess_ufunc_result(self, result: jax.Array, *, ufunc: jnp.ufunc, method: str) -> jax.Array:
        del ufunc, method
        return jnp.maximum(result, 1e-10)

    @override
    def __eq__(self, value: Any) -> jax.Array: # ty: ignore[invalid-method-override] # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, JaxDirichletDistribution):
            eq = jnp.equal(self.alphas, value.alphas)
        else:
            eq = jnp.equal(self.alphas, value)
        return jnp.all(eq, axis=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash.
        
        We intentionally bypass ``super()`` here because protocol-heave MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance indentity semantics.
        """
        return object.__hash__(self)