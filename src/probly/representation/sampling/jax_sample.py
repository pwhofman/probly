"""JAX sample implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, override

import jax
import jax.numpy as jnp
import numpy as np

from probly.representation.sampling.common_sample import Sample, SampleAxis, create_sample

if TYPE_CHECKING:
    from collections.abc import Iterable

    from jax import Device
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    import numpy.typing as npt


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxArraySample(Sample[jax.Array]):
    """A sample implementation for JAX arrays."""

    array: jax.Array
    sample_axis: int

    def __post_init__(self) -> None:
        """Validate the sample_axis."""
        if self.sample_axis >= self.array.ndim:
            msg = f"sample_axis {self.sample_axis} out of bounds for array with ndim {self.array.ndim}."
            raise ValueError(msg)
        if self.sample_axis < 0:
            if self.sample_axis < -self.array.ndim:
                msg = f"sample_axis {self.sample_axis} out of bounds for array with ndim {self.array.ndim}."
                raise ValueError(msg)
            super().__setattr__("sample_axis", self.array.ndim + self.sample_axis)

        if not isinstance(self.array, jax.Array):
            msg = "array must be a JAX array."
            raise TypeError(msg)

    @classmethod
    def from_iterable(
        cls,
        samples: Iterable[jax.Array],
        sample_axis: SampleAxis = "auto",
        dtype: DTypeLike | None = None,
    ) -> Self:
        """Create an JaxArraySample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            sample_axis: The dimension along which samples are organized.
            dtype: Desired data type of the array.

        Returns:
            The created JaxArraySample.
        """
        if isinstance(samples, jax.Array):
            if sample_axis == "auto":
                if samples.ndim == 0:
                    msg = "Cannot infer sample_axis for 0-dimensional array."
                    raise ValueError(msg)
                sample_axis = 0 if samples.ndim == 1 else 1
            if sample_axis != 0:
                samples = jnp.moveaxis(samples, 0, sample_axis)
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
                sample_axis = (
                    (0 if first_sample.ndim == 0 else 1) if isinstance(first_sample, (np.ndarray, jax.Array)) else 0
                )
            samples = jnp.stack(samples, axis=sample_axis, dtype=dtype)

        return cls(array=samples, sample_axis=sample_axis)

    @override
    @classmethod
    def from_sample(
        cls,
        sample: Sample[jax.Array],
        sample_axis: SampleAxis = "auto",
        dtype: DTypeLike | None = None,
    ) -> Self:
        if isinstance(sample, JaxArraySample):
            sample_array = sample.array

            if dtype is not None:
                sample_array = sample_array.astype(dtype)

            in_sample_axis = sample.sample_axis
            if sample_axis not in ("auto", in_sample_axis):
                sample_array = jnp.moveaxis(sample_array, in_sample_axis, sample_axis)  # type: ignore[arg-type]
                in_sample_axis = sample_axis  # type: ignore[assignment]
            return cls(array=sample_array, sample_axis=in_sample_axis)

        return cls.from_iterable(sample.samples, sample_axis=sample_axis, dtype=dtype)

    def __len__(self) -> int:
        """Return the len of the array."""
        return len(self.array)

    def __array_namespace__(self) -> Any:  # noqa: ANN401
        """Get the array namespace of the underlying array."""
        return self.array.__array_namespace__()

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the underlying array."""
        return self.array.dtype

    @property
    def device(self) -> Any:  # noqa: ANN401
        """The device of the underlying array."""
        return self.array.device

    @property
    def mT(self) -> jax.Array:  # noqa: N802
        """The transposed version of the underlying array."""
        return jnp.matrix_transpose(self.array)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the underlying array."""
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying array."""
        return self.array.shape

    @property
    def size(self) -> int:
        """The total number of elements in the underlying array."""
        return self.array.size

    @property
    def T(self) -> jax.Array:  # noqa: N802
        """The transposed version of the underlying array."""
        return jnp.transpose(self.array)

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return self.array.shape[self.sample_axis]

    @property
    def samples(self) -> jax.Array:
        """Return an iterator over the samples."""
        if self.sample_axis == 0:
            return self.array
        return jnp.moveaxis(self.array, self.sample_axis, 0)

    def sample_mean(self) -> jax.Array:
        """Compute the mean of the sample."""
        return jnp.mean(self.array, axis=self.sample_axis)

    def sample_std(self, ddof: int = 1) -> jax.Array:
        """Compute the standard deviation of the sample."""
        return jnp.std(self.array, axis=self.sample_axis, ddof=ddof)

    def sample_var(self, ddof: int = 1) -> jax.Array:
        """Compute the variance of the sample."""
        return jnp.var(self.array, axis=self.sample_axis, ddof=ddof)

    @override
    def concat(self, other: Sample[jax.Array]) -> Self:
        if isinstance(other, JaxArraySample):
            other_array = jnp.moveaxis(other.array, other.sample_axis, self.sample_axis)
        else:
            other_array = jnp.stack(list(other.samples), axis=self.sample_axis, dtype=self.array.dtype)

        concatenated = jnp.concatenate((self.array, other_array), axis=self.sample_axis)

        return type(self)(array=concatenated, sample_axis=self.sample_axis)

    def move_sample_axis(self, new_sample_axis: int) -> JaxArraySample:
        """Return a new JaxArraySample with the sample dimension moved to new_sample_axis.

        Args:
            new_sample_axis: The new sample dimension.

        Returns:
            A new ArraySample with the sample dimension moved.
        """
        moved_array = jnp.moveaxis(self.array, self.sample_axis, new_sample_axis)
        return type(self)(array=moved_array, sample_axis=new_sample_axis)

    def __array__(self, dtype: npt.DTypeLike = None, copy: bool | None = None) -> np.ndarray:
        """Get the underlying numpy array.

        Args:
            dtype: Desired data type of the array.
            copy: Whether to return a copy of the array.

        Returns:
            The underlying numpy array.
        """
        return np.asarray(self.array, dtype=dtype, copy=copy)

    def copy(self) -> Self:
        """Create a copy of the JaxArraySample.

        Returns:
            A copy of the JaxArraySample.
        """
        return type(self)(array=self.array.copy(), sample_axis=self.sample_axis)

    def to_device(
        self,
        device: Device | Sharding,  # pyright: ignore[reportInvalidTypeForm]
        *,
        stream: int | Any | None = None,  # noqa: ANN401
    ) -> Self:
        """Move the underlying array to the specified device.

        Args:
            device: The target device.
            stream: not implemented, passing a non-None value will lead to an error.

        Returns:
            A new JaxArraySample on the specified device.
        """
        if stream is not None:
            msg = "stream argument of array.to_device()"
            raise NotImplementedError(msg)

        if device == self.device:
            return self

        return type(self)(array=self.array.to_device(device), sample_axis=self.sample_axis)


create_sample.register(jax.Array, JaxArraySample.from_iterable)
