"""Jax-based distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

import jax
from jax import numpy as jnp
from jax.core import Tracer

from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.distribution._common import DirichletDistribution, create_dirichlet_distribution_from_alphas
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.jax_functions import jax_add, jax_average, jax_mean, jax_subtract, jax_sum
from probly.representation.sample.jax import JaxArraySample
from probly.utils.jax import fresh_prng_key

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax._src.typing import ArrayLike
    from jax.typing import DTypeLike


@create_dirichlet_distribution_from_alphas.register(jax.Array)
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
    permitted_functions: ClassVar[set[Callable]] = {jax_mean, jax_sum, jax_average, jax_add, jax_subtract}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.alphas, jax.Array):
            msg = "alphas must be a jax Array."
            raise TypeError(msg)

        if self.alphas.ndim < 1:
            msg = "alphas must have at least one dimension."
            raise ValueError(msg)

        # Reconstruction during tracing can validate shapes, but not array values.
        if not isinstance(self.alphas, Tracer) and jnp.any(self.alphas <= 0):
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
            prng_key = fresh_prng_key()

        gammas = jax.random.gamma(
            prng_key,
            self.alphas,
            shape=(num_samples, *self.alphas.shape),
        )
        return JaxArraySample(array=JaxProbabilityCategoricalDistribution(gammas), sample_axis=0)

    @override
    def _postprocess_elementwise_result(
        self, values: dict[str, Any], *, func: Callable, operands: tuple[object, ...]
    ) -> dict[str, Any]:
        """Keep concentration parameters strictly positive after ``+``/``-``."""
        del operands
        if func in (jax_add, jax_subtract):
            return {name: jnp.maximum(value, 1e-10) for name, value in values.items()}
        return values

    @override
    def __eq__(self, value: Any) -> jax.Array:  # ty: ignore[invalid-method-override] # noqa: PYI032
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
