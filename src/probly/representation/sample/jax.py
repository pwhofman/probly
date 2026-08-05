# pyright: reportInvalidTypeForm=false
"""JAX sample implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast, override

import jax
import jax.numpy as jnp
import numpy as np

from probly.representation.array_like import ToIndices, to_numpy_array_like
from probly.representation.jax_functions import (
    jax_average,
    jax_concatenate,
    jax_mean,
    jax_moveaxis,
    jax_stack,
    jax_std,
    jax_var,
)
from probly.representation.jax_like import JaxLike, JaxLikeImplementation
from probly.representation.sample._common import Sample, SampleAxis, create_sample
from probly.representation.sample.array import ArraySample
from probly.representation.sample.axis_tracking import track_axis
from probly.representation.sample.jax_functions import JaxSampleInternals, jax_function, jax_sample_internals

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from jax import Device
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    import numpy.typing as npt


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxArraySample[D: JaxLike | jax.Array](JaxLikeImplementation[D], Sample[D]):
    """A sample implementation for JAX arrays."""

    array: D
    sample_axis: int
    weights: jax.Array | None = None

    def __post_init__(self) -> None:
        """Validate the sample_axis."""
        if self.sample_axis >= self.array.ndim:
            msg = f"sample_axis {self.sample_axis} out of bounds for array with ndim {self.array.ndim}."
            raise ValueError(msg)
        if self.sample_axis < 0:
            if self.sample_axis < -self.array.ndim:
                msg = f"sample_axis {self.sample_axis} out of bounds for array with ndim {self.array.ndim}."
                raise ValueError(msg)
            super(type(self), self).__setattr__("sample_axis", self.array.ndim + self.sample_axis)

        if not isinstance(self.array, (jax.Array, JaxLikeImplementation)):
            msg = "array must be a JAX array."
            raise TypeError(msg)

        if self.weights is not None and self.weights.shape != (self.sample_size,):
            msg = f"weights must have shape ({self.sample_size},) but got {self.weights.shape}."
            raise ValueError(msg)

    @override
    @classmethod
    def from_iterable(
        cls,
        samples: Iterable[D],
        weights: Iterable[float] | None = None,
        sample_axis: SampleAxis = "auto",
        dtype: DTypeLike | None = None,
    ) -> Self:
        """Create an JaxArraySample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            weights: Optional weights for the samples.
            sample_axis: The dimension along which samples are organized.
            dtype: Desired data type of the array.

        Returns:
            The created JaxArraySample.
        """
        if isinstance(samples, JaxLike):
            sample_array = cast("Any", samples)
            if sample_axis == "auto":
                if sample_array.ndim == 0:
                    msg = "Cannot infer sample_axis for 0-dimensional array."
                    raise ValueError(msg)
                sample_axis = -1
            if sample_axis != 0:
                sample_array = jax_moveaxis(sample_array, 0, sample_axis)
            if dtype is not None:
                sample_array = sample_array.astype(dtype)
        else:
            if not isinstance(samples, Sequence):
                samples = list(samples)
            if sample_axis == "auto":
                if len(samples) == 0:
                    msg = "Cannot infer sample_axis for empty samples."
                    raise ValueError(msg)
                sample_axis = -1
            sample_array = jax_stack(cast("Any", samples), axis=sample_axis, dtype=dtype)

        return cls(
            array=cast("D", sample_array),
            sample_axis=sample_axis,
            weights=jnp.asarray(weights) if weights is not None else None,
        )

    @override
    @classmethod
    def from_sample(
        cls,
        sample: Sample[D],
        sample_axis: SampleAxis = "auto",
        dtype: DTypeLike | None = None,
    ) -> Self:
        if isinstance(sample, JaxArraySample):
            sample_array: D = sample.array  # ty: ignore[invalid-assignment]
            sample_weights = sample.weights

            if dtype is not None:
                sample_array = cast("Any", sample_array).astype(dtype)

            in_sample_axis: int = sample.sample_axis
            if sample_axis not in ("auto", in_sample_axis):
                sample_array = cast("D", jax_moveaxis(cast("Any", sample_array), in_sample_axis, sample_axis))
                in_sample_axis = sample_axis
            return cls(array=sample_array, sample_axis=in_sample_axis, weights=sample_weights)

        return cls.from_iterable(sample.samples, weights=sample.weights, sample_axis=sample_axis, dtype=dtype)

    def __len__(self) -> int:
        """Return the len of the array."""
        return len(self.array)

    @override
    def __iter__(self) -> Iterator[Any]:
        """Iterate over axis 0 of the sample wrapper."""
        for index in range(len(self)):
            yield self[index]

    @override
    def __array_namespace__(self, /, *, api_version: str | None = None) -> Any:
        """Get the array namespace of the underlying array.

        Args:
            api_version: The requested version of the array API standard.

        Returns:
            The array namespace of the underlying array.
        """
        del api_version
        return cast("Any", self.array).__array_namespace__()

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the underlying array."""
        return self.array.dtype

    @property
    def device(self) -> Any:  # noqa: ANN401
        """The device of the underlying array."""
        return self.array.device

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
    def at(self) -> Any:  # noqa: ANN401
        """The indexed update helper of the underlying array."""
        return self.array.at

    @override
    def block_until_ready(self) -> Self:
        """Block until the asynchronous computation of the underlying arrays has finished.

        Returns:
            The sample itself.
        """
        cast("Any", self.array).block_until_ready()
        if self.weights is not None:
            self.weights.block_until_ready()
        return self

    @override
    def astype(
        self,
        dtype: DTypeLike | None,
        copy: bool = False,
        device: Device | Sharding | None = None,
    ) -> Self:
        """Cast the underlying array to a new data type.

        Args:
            dtype: The target data type.
            copy: Whether to always return a copy.
            device: The device the result should live on.

        Returns:
            A new JaxArraySample with the cast array.
        """
        return type(self)(
            array=cast("D", cast("Any", self.array).astype(dtype, copy=copy, device=device)),
            sample_axis=self.sample_axis,
            weights=self.weights,
        )

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return self.array.shape[self.sample_axis]

    @property
    def samples(self) -> D:
        """Return an iterator over the samples."""
        if self.sample_axis == 0:
            return self.array
        return cast("D", jax_moveaxis(cast("Any", self.array), self.sample_axis, 0))

    def sample_mean(self) -> D:
        """Compute the mean of the sample."""
        array = cast("Any", self.array)
        if self.weights is not None:
            return cast("D", jax_average(array, self.sample_axis, self.weights))

        return cast("D", jax_mean(array, self.sample_axis))

    def sample_std(self, ddof: int = 0) -> D:
        """Compute the standard deviation of the sample."""
        if self.weights is not None:
            return cast("D", jnp.sqrt(cast("Any", self.sample_var(ddof=ddof))))

        return cast("D", jax_std(cast("Any", self.array), self.sample_axis, ddof=ddof))

    def sample_var(self, ddof: int = 0) -> D:
        """Compute the variance of the sample."""
        array = cast("Any", self.array)
        weights = self.weights
        if weights is not None:
            if ddof != 0:
                msg = "Weighted samples do not support ddof > 0."
                raise ValueError(msg)
            average = jax_average(array, self.sample_axis, weights, keepdims=True)
            return cast("D", jax_average((array - average) ** 2, self.sample_axis, weights))

        return cast("D", jax_var(array, self.sample_axis, ddof=ddof))

    @override
    def concat(self, other: Sample[D]) -> Self:
        if isinstance(other, JaxArraySample):
            other_array = jax_moveaxis(cast("Any", other.array), other.sample_axis, self.sample_axis)
        else:
            other_array = jax_stack(
                cast("Any", list(other.samples)), self.sample_axis, dtype=cast("Any", self.array).dtype
            )

        concatenated = jax_concatenate((cast("Any", self.array), other_array), self.sample_axis)

        weights = self.weights
        other_weights = other.weights

        if weights is not None or other_weights is not None:
            if weights is None:
                weights = jnp.ones(self.sample_size)
            other_weights = jnp.ones(other.sample_size) if other_weights is None else jnp.asarray(other_weights)
            weights = jax_concatenate((weights, other_weights), 0)

        return type(self)(array=cast("D", concatenated), sample_axis=self.sample_axis, weights=weights)

    def move_sample_axis(self, new_sample_axis: int) -> JaxArraySample[D]:
        """Return a new JaxArraySample with the sample dimension moved to new_sample_axis.

        Args:
            new_sample_axis: The new sample dimension.

        Returns:
            A new ArraySample with the sample dimension moved.
        """
        moved_array = jax_moveaxis(cast("Any", self.array), self.sample_axis, new_sample_axis)
        return type(self)(array=cast("D", moved_array), sample_axis=new_sample_axis, weights=self.weights)

    def __getitem__(self, index: ToIndices) -> Any:  # noqa: ANN401
        """Get a sample by index.

        Args:
            index: The index to select with.

        Returns:
            A new JaxArraySample if the sample axis survives the indexing operation, otherwise
            the plain indexed array.

        Raises:
            IndexError: If the sample is weighted and the index cannot be applied to the weights.
        """
        new_array = cast("Any", self.array)[cast("Any", index)]

        if not hasattr(new_array, "ndim"):
            return new_array

        track_result = track_axis(index, self.sample_axis, self.array.ndim, torch_indexing=False)

        if track_result is None:
            return new_array

        weights = self.weights

        if weights is not None:
            weights_index = track_result.index
            if weights_index is NotImplemented:
                msg = "Weighted samples do not support this indexing operation."
                raise IndexError(msg)
            weights = weights[cast("Any", weights_index)]

        return type(self)(array=new_array, sample_axis=track_result.new_axis, weights=weights)

    def __setitem__(self, index: ToIndices, value: object) -> None:
        """Reject in-place assignment, JAX arrays are immutable.

        Args:
            index: The index that was assigned to.
            value: The value that was assigned.

        Raises:
            TypeError: Always, use :attr:`at` for indexed updates instead.
        """
        del index, value
        msg = "JAX arrays are immutable, use sample.at[index].set(value) instead."
        raise TypeError(msg)

    @override
    def __array__(self, dtype: npt.DTypeLike | None = None, /, *, copy: bool | None = None) -> np.ndarray:
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
        return type(self)(
            array=cast("D", cast("Any", self.array).copy()),
            sample_axis=self.sample_axis,
            weights=self.weights.copy() if self.weights is not None else None,
        )

    @override
    def to_device(
        self,
        device: Device | Sharding,
        /,
        *,
        stream: int | Any | None = None,
    ) -> Self:
        """Move the underlying array to the specified device.

        Args:
            device: The target device.
            stream: not implemented, passing a non-None value will lead to an error.

        Returns:
            A new JaxArraySample on the specified device.

        Raises:
            NotImplementedError: If a stream is given.
        """
        if stream is not None:
            msg = "stream argument of array.to_device()"
            raise NotImplementedError(msg)

        if device == self.device:
            return self

        return type(self)(
            array=cast("D", cast("Any", self.array).to_device(device)),
            sample_axis=self.sample_axis,
            weights=self.weights.to_device(device) if self.weights is not None else None,
        )

    @classmethod
    @override
    def __jax_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Handle the jax function wrappers.

        Args:
            func: The wrapper function that was called.
            types: The types of the overriding arguments.
            args: The positional arguments of the call.
            kwargs: The keyword arguments of the call.

        Returns:
            The result of the call, or ``NotImplemented`` if it is not supported.
        """
        del cls
        return jax_function(func, types, args, {} if kwargs is None else kwargs)

    def __jax_like__(
        self,
        dtype: DTypeLike | None = None,
        /,
        *,
        device: Device | Sharding | None = None,
        copy: bool = False,
    ) -> JaxLikeImplementation[Any]:
        """Convert to a JaxLike.

        Args:
            dtype: The desired data type of the underlying array.
            device: The desired device of the underlying array.
            copy: Whether to always return a copy.

        Returns:
            A JaxArraySample with the requested data type and device.
        """
        if dtype is None and device is None and not copy:
            return self

        return type(self)(
            array=cast("D", jnp.asarray(self.array, dtype=dtype, device=device, copy=copy)),
            sample_axis=self.sample_axis,
            weights=jnp.asarray(self.weights, device=device, copy=copy) if self.weights is not None else None,
        )

    def __array_like__(self, dtype: npt.DTypeLike | None = None, /, *, copy: bool | None = None) -> ArraySample[Any]:
        """Convert to a NumpyArrayLike.

        Args:
            dtype: The desired data type of the underlying array.
            copy: Whether to return a copy of the underlying array.

        Returns:
            An ArraySample wrapping the converted array.
        """
        array = to_numpy_array_like(self.array, dtype=dtype, copy=copy)

        return ArraySample(
            cast("Any", array),
            sample_axis=self.sample_axis,
            weights=np.asarray(self.weights) if self.weights is not None else None,
        )


@jax_sample_internals.register(JaxArraySample)
def _(sample: JaxArraySample) -> JaxSampleInternals[jax.Array]:
    """Get internals for a JaxArraySample."""
    return JaxSampleInternals[jax.Array](
        create=type(sample),
        array=sample.array,
        sample_axis=sample.sample_axis,
        weights=sample.weights,
    )


create_sample.register(
    jax.Array | JaxLikeImplementation,
    JaxArraySample.from_iterable,
)
