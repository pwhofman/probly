"""Numpy-based sample representation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, override

import numpy as np

from probly.representation.sampling.array_sample_axis_tracking import Index, track_axis
from probly.representation.sampling.array_sample_functions import (
    array_function,
    array_sample_internals,
    track_sample_axis_after_reduction,
)
from probly.representation.sampling.common_sample import Sample, SampleAxis, create_sample

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import DTypeLike

type Numeric = np.number | np.ndarray | float | int


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArraySample[D: Numeric](Sample[D], np.lib.mixins.NDArrayOperatorsMixin):
    """A sample of predictions stored in a numpy array."""

    array: np.ndarray
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

        if not isinstance(self.array, np.ndarray):
            msg = "array must be a numpy ndarray."
            raise TypeError(msg)

    @classmethod
    def from_iterable(cls, samples: Iterable[D], sample_axis: SampleAxis = "auto", dtype: DTypeLike = None) -> Self:
        """Create an ArraySample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            sample_axis: The dimension along which samples are organized.
            dtype: Desired data type of the array.

        Returns:
            The created ArraySample.
        """
        if isinstance(samples, np.ndarray):
            if sample_axis == "auto":
                if samples.ndim == 0:
                    msg = "Cannot infer sample_axis for 0-dimensional array."
                    raise ValueError(msg)
                sample_axis = 0 if samples.ndim == 1 else 1
            if sample_axis != 0:
                samples = np.moveaxis(samples, 0, sample_axis)
            if dtype is not None:
                samples = samples.astype(dtype)
        else:
            if hasattr(samples, "__array__"):
                return cls.from_iterable(np.asarray(samples, dtype=dtype), sample_axis=sample_axis)
            if not isinstance(samples, Sequence):
                samples = list(samples)
            if sample_axis == "auto":
                if len(samples) == 0:
                    msg = "Cannot infer sample_axis for empty samples."
                    raise ValueError(msg)
                first_sample = samples[0]
                sample_axis = (0 if first_sample.ndim == 0 else 1) if isinstance(first_sample, np.ndarray) else 0
            samples = np.stack(samples, axis=sample_axis, dtype=dtype)

        return cls(array=samples, sample_axis=sample_axis)

    @override
    @classmethod
    def from_sample(cls, sample: Sample[D], sample_axis: SampleAxis = "auto", dtype: DTypeLike = None) -> Self:
        if isinstance(sample, ArraySample):
            sample_array = sample.array

            if dtype is not None:
                sample_array = sample_array.astype(dtype)

            in_sample_axis = sample.sample_axis
            if sample_axis not in ("auto", in_sample_axis):
                sample_array = np.moveaxis(sample_array, in_sample_axis, sample_axis)  # type: ignore[arg-type]
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
        return self.array.dtype  # type: ignore[no-any-return]

    @property
    def device(self) -> str:
        """The device of the underlying array."""
        return self.array.device

    @property
    def mT(self) -> Self:  # noqa: N802
        """The transposed version of the underlying array."""
        return np.matrix_transpose(self)  #  type: ignore[return-value]

    @property
    def ndim(self) -> int:
        """The number of dimensions of the underlying array."""
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying array."""
        return self.array.shape  # type: ignore[no-any-return]

    @property
    def size(self) -> int:
        """The total number of elements in the underlying array."""
        return self.array.size

    @property
    def T(self) -> Self:  # noqa: N802
        """The transposed version of the underlying array."""
        return np.transpose(self)  #  type: ignore[return-value]

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return self.array.shape[self.sample_axis]  # type: ignore[no-any-return]

    @property
    def samples(self) -> np.ndarray:
        """Return an iterator over the samples."""
        if self.sample_axis == 0:
            return self.array
        return np.moveaxis(self.array, self.sample_axis, 0)

    @override
    def sample_mean(self) -> D:
        """Compute the mean of the sample."""
        return self.array.mean(axis=self.sample_axis)  # type: ignore[no-any-return]

    @override
    def sample_std(self, ddof: int = 1) -> D:
        """Compute the standard deviation of the sample."""
        return self.array.std(axis=self.sample_axis, ddof=ddof)  # type: ignore[no-any-return]

    @override
    def sample_var(self, ddof: int = 1) -> D:
        """Compute the variance of the sample."""
        return self.array.var(axis=self.sample_axis, ddof=ddof)  # type: ignore[no-any-return]

    @override
    def concat(self, other: Sample[D]) -> Self:
        if isinstance(other, ArraySample):
            other_array = np.moveaxis(other.array, other.sample_axis, self.sample_axis)
        else:
            other_array = np.stack(list(other.samples), axis=self.sample_axis, dtype=self.array.dtype)

        concatenated = np.concatenate((self.array, other_array), axis=self.sample_axis)

        return type(self)(array=concatenated, sample_axis=self.sample_axis)

    def move_sample_axis(self, new_sample_axis: int) -> ArraySample[D]:
        """Return a new ArraySample with the sample dimension moved to new_sample_axis.

        Args:
            new_sample_axis: The new sample dimension.

        Returns:
            A new ArraySample with the sample dimension moved.
        """
        moved_array = np.moveaxis(self.array, self.sample_axis, new_sample_axis)
        return type(self)(array=moved_array, sample_axis=new_sample_axis)

    def __getitem__(self, index: Index) -> Self | D | np.ndarray:
        """Get a sample by index.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            The sample at the specified index.
        """
        new_array = self.array[index]

        if not isinstance(new_array, np.ndarray):
            return new_array

        new_sample_axis = track_axis(index, self.sample_axis, self.array.ndim)

        if new_sample_axis is None:
            return new_array

        return type(self)(array=new_array, sample_axis=new_sample_axis)

    def __setitem__(self, index: int | slice | np.ndarray, value: D | np.ndarray) -> None:
        """Set a sample by index.

        Args:
            index: The index of the sample to set.
            value: The value to set at the specified index.
        """
        self.array[index] = value

    def __array__(self, dtype: DTypeLike = None, copy: bool | None = None) -> np.ndarray:
        """Get the underlying numpy array.

        Args:
            dtype: Desired data type of the array.
            copy: Whether to return a copy of the array.

        Returns:
            The underlying numpy array.
        """
        if dtype is None and not copy:
            return self.array

        return np.asarray(self.array, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Handle numpy ufuncs.

        Args:
            ufunc: The ufunc to apply.
            method: The method of the ufunc.
            inputs: The input arrays.
            kwargs: Additional keyword arguments.

        Returns:
            The result of applying the ufunc.
        """
        new_sample_axis: int | None = self.sample_axis
        arrays = [x.array if isinstance(x, ArraySample) else x for x in inputs]

        if method in ("__call__", "reduce", "reduceat", "accumulate") and "out" in kwargs:
            outs = kwargs["out"]
            if outs is not None:
                if not isinstance(outs, tuple):
                    outs = (outs,)
                cast_outs = tuple(o.array if isinstance(o, ArraySample) else o for o in outs)
                kwargs["out"] = cast_outs
        else:
            outs = None

        result = getattr(ufunc, method)(*arrays, **kwargs)

        if method in {"reduce", "reduceat", "accumulate"}:
            axis: int | tuple[int, ...] = kwargs.get("axis", 0)
            keepdims: bool = method == "reduce" and kwargs.get("keepdims", False)
            new_sample_axis = track_sample_axis_after_reduction(self.sample_axis, self.ndim, axis, keepdims)
        elif method in ("at", "outer"):
            new_sample_axis = None

        if outs is not None:
            if len(outs) == 1:
                return outs[0]
            return outs

        if new_sample_axis is not None and isinstance(result, np.ndarray):
            return type(self)(result, sample_axis=new_sample_axis)
        return result

    def __array_function__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Handle numpy array functions.

        Args:
            func: The numpy function to apply.
            types: The types of the input arguments.
            args: The input arguments.
            kwargs: Additional keyword arguments.

        Returns:
            The result of applying the numpy function.
        """
        return array_function(
            func,
            types,
            args,
            kwargs,
        )

    def copy(self) -> Self:
        """Create a copy of the ArraySample.

        Returns:
            A copy of the ArraySample.
        """
        return type(self)(array=self.array.copy(), sample_axis=self.sample_axis)

    def to_device(self, device: Literal["cpu"]) -> Self:
        """Move the underlying array to the specified device.

        Args:
            device: The target device.

        Returns:
            A new ArraySample on the specified device.
        """
        if device == self.device:
            return self

        return type(self)(array=self.array.to_device(device), sample_axis=self.sample_axis)

    def __eq__(self, value: Any) -> Self:  # type: ignore[override]  # noqa: ANN401, PYI032
        """Vectorized equality comparison."""
        return np.equal(self, value)  # type: ignore[return-value]

    def __hash__(self) -> int:
        """Compute the hash of the ArraySample."""
        return super().__hash__()


@array_sample_internals.register
def _(array: ArraySample) -> tuple[np.ndarray, int]:
    """Get the sample dimension of an ArraySample."""
    return array.array, array.sample_axis


create_sample.register(np.number | np.ndarray | float | int, ArraySample.from_iterable)
