"""JAX sample implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .sample import Sample, create_sample


@create_sample.register(jax.Array)
class JaxArraySample(Sample[jax.Array]):
    """A sample implementation for JAX arrays."""

    def __init__(self, samples: list[jax.Array]) -> None:
        """Initialize the JAX array sample."""
        self.array = jnp.stack(samples).transpose(1, 0, 2)  # we use the convention [instances, samples, classes]

    def mean(self) -> jax.Array:
        """Compute the mean of the sample."""
        return jnp.mean(self.array, axis=1)

    def std(self, ddof: int = 1) -> jax.Array:
        """Compute the standard deviation of the sample."""
        return jnp.std(self.array, axis=1, ddof=ddof)

    def var(self, ddof: int = 1) -> jax.Array:
        """Compute the variance of the sample."""
        return jnp.var(self.array, axis=1, ddof=ddof)
