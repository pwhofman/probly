"""JAX sample implementation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from probly.representation.sampling.common_sample import Sample

from .sample import create_sample

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import DTypeLike

    from probly.representation.sampling.common_sample import SampleAxis

type Numeric = jax.numpy.number | jax.Array | float | int


@create_sample.register(jax.Array)
class JaxArraySample(Sample[jax.Array]):
    """A sample implementation for JAX arrays."""

    sample_axis: int
    array: jax.Array

    def __init__(self, samples: list[jax.Array], sample_axis: int = 1) -> None:
        """Initialize the JAX array sample."""
        self.array = jnp.stack(samples).transpose(1, 0, 2)  # we use the convention [instances, samples, classes]
        self.sample_axis = sample_axis

    def from_iterable(
        self,
        samples: Iterable[Numeric],
        sample_axis: SampleAxis = "auto",
        dtype: DTypeLike = None,
    ) -> JaxArraySample:
        """Create an JaxArraySample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            sample_axis: The dimension along which samples are organized.
            dtype: Desired data type of the array.

        Returns:
            The created ArraySample.
        """
        if isinstance(samples, jax.Array):
            if sample_axis == "auto":
                if samples.ndim == 0:
                    msg = "Cannot infer sample_axis for 0-dimensional array."
                    raise ValueError(msg)
                sample_axis = 0 if samples.ndim == 1 else 1
            if sample_axis != 0:
                samples = jax.numpy.moveaxis(samples, 0, sample_axis)
            if dtype is not None:
                samples = samples.astype(dtype)
        else:
            if not isinstance(samples, Sequence):
                samples = list(samples)
            if sample_axis == "auto":
                if len(samples) == 0:
                    msg = "Cannot infer sample_axis for empty samples."
                    raise ValueError(msg)
                first_sample = samples[0]
                sample_axis = (0 if first_sample.ndim == 0 else 1) if isinstance(first_sample, jax.Array) else 0
            samples = jax.numpy.stack(samples, axis=sample_axis, dtype=dtype)

        return self(array=samples, sample_axis=sample_axis)

    def samples(self) -> jax.Array:
        """Return an iterator over the samples."""
        if self.sample_axis == 0:
            return self.array
        return jax.numpy.moveaxis(self.array, self.sample_axis, 0)

    def sample_mean(self) -> jax.Array:
        """Compute the mean of the sample."""
        return jnp.mean(self.array, axis=self.sample_axis)

    def sample_std(self, ddof: int = 1) -> jax.Array:
        """Compute the standard deviation of the sample."""
        return jnp.std(self.array, axis=self.sample_axis, ddof=ddof)

    def sample_var(self, ddof: int = 1) -> jax.Array:
        """Compute the variance of the sample."""
        return jnp.var(self.array, axis=self.sample_axis, ddof=ddof)
