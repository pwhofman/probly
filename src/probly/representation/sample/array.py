"""Numpy-based sample representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, Unpack, cast, override

import numpy as np

from probly.representation.array_like import (
    ArrayFlagsLike,
    NumpyArrayLike,
    NumpyArrayLikeConvertible,
    NumpyArrayLikeImplementation,
    Order,
    ToIndices,
    to_numpy_array_like,
)
from probly.representation.sample._common import Sample, SampleAxis, SampleParams, create_sample
from probly.representation.sample.array_functions import (
    ArraySampleInternals,
    array_function,
    array_sample_internals,
    track_sample_axis_after_reduction,
)
from probly.representation.sample.axis_tracking import track_axis

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from types import ModuleType

    from numpy.typing import DTypeLike
    import torch

    from probly.representation.torch_like import TorchTensorLikeImplementation


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArraySample[D: NumpyArrayLike | np.ndarray](NumpyArrayLikeImplementation[D], Sample[D]):
    """A sample of predictions stored in a numpy array."""

    array: D
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
            super(type(self), self).__setattr__("sample_axis", self.array.ndim + self.sample_axis)

        if not isinstance(self.array, NumpyArrayLike):
            msg = "array must be a NumpyArrayLike (or ndarray)."
            raise TypeError(msg)

    @override
    @classmethod
    def from_iterable(
        cls,
        samples: Iterable[D],
        sample_axis: SampleAxis = "auto",
        dtype: DTypeLike | None = None,
        **_kwargs: Unpack[SampleParams],
    ) -> Self:
        """Create an ArraySample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            sample_axis: The dimension along which samples are organized.
            dtype: Desired data type of the array.

        Returns:
            The created ArraySample.
        """
        if isinstance(samples, NumpyArrayLike):
            sample_array = to_numpy_array_like(samples, dtype=dtype)
            if sample_axis == "auto":
                if sample_array.ndim == 0:
                    msg = "Cannot infer sample_axis for 0-dimensional array."
                    raise ValueError(msg)
                sample_axis = -1
            if sample_axis != 0:
                samples = np.moveaxis(sample_array, 0, sample_axis)  # ty:ignore[invalid-argument-type]
        else:
            samples = [to_numpy_array_like(s, dtype=dtype) for s in samples]  # ty:ignore[invalid-assignment]
            if sample_axis == "auto":
                if len(samples) == 0:  # ty:ignore[invalid-argument-type]
                    msg = "Cannot infer sample_axis for empty samples."
                    raise ValueError(msg)
                sample_axis = -1
            samples = np.stack(samples, axis=sample_axis, dtype=dtype)  # ty:ignore[no-matching-overload]

        return cls(array=samples, sample_axis=sample_axis)

    @override
    @classmethod
    def from_sample(cls, sample: Sample[D], sample_axis: SampleAxis = "auto", dtype: DTypeLike | None = None) -> Self:  # ty:ignore[invalid-method-override]
        if isinstance(sample, NumpyArrayLikeConvertible):
            array_sample = to_numpy_array_like(sample, dtype=dtype)
            if not isinstance(array_sample, ArraySample):
                msg = "Converted array must be an ArraySample."
                raise TypeError(msg)
            sample = array_sample

        if isinstance(sample, ArraySample):
            sample_array: D = sample.array  # ty:ignore[invalid-assignment]

            if dtype is not None:
                sample_array = cast("Any", sample_array).astype(dtype)

            in_sample_axis = sample.sample_axis
            if sample_axis not in ("auto", in_sample_axis):
                sample_array = np.moveaxis(sample_array, in_sample_axis, sample_axis)  # ty:ignore[invalid-argument-type]
                in_sample_axis = sample_axis
            return cls(array=sample_array, sample_axis=in_sample_axis)

        return cls.from_iterable(sample.samples, sample_axis=sample_axis, dtype=dtype)

    @override
    def __len__(self) -> int:
        """Return the len of the array."""
        return len(cast("Any", self.array))

    def __array_namespace__(
        self, /, *, api_version: Literal["2022.12", "2023.12", "2024.12"] | None = None
    ) -> ModuleType:
        """Get the array namespace of the underlying array."""
        return cast("Any", self.array).__array_namespace__(api_version=api_version)

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the underlying array."""
        return self.array.dtype

    @property
    def device(self) -> str:
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

    @override
    @property
    def flags(self) -> ArrayFlagsLike:
        return self.array.flags

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return self.array.shape[self.sample_axis]

    @property
    def samples(self) -> D:
        """Return an iterator over the samples."""
        if self.sample_axis == 0:
            return self.array
        return np.moveaxis(self.array, self.sample_axis, 0)  # ty:ignore[invalid-argument-type]

    @override
    def sample_mean(self) -> D:
        """Compute the mean of the sample."""
        return np.mean(self.array, axis=self.sample_axis)

    @override
    def sample_std(self, ddof: int = 1) -> D:
        """Compute the standard deviation of the sample."""
        return np.std(self.array, axis=self.sample_axis, ddof=ddof)

    @override
    def sample_var(self, ddof: int = 1) -> D:
        """Compute the variance of the sample."""
        return np.var(self.array, axis=self.sample_axis, ddof=ddof)

    @override
    def concat(self, other: Sample[D]) -> Self:
        if isinstance(other, ArraySample):
            other_array = np.moveaxis(other.array, other.sample_axis, self.sample_axis)  # ty:ignore[invalid-argument-type]
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
        moved_array = np.moveaxis(self.array, self.sample_axis, new_sample_axis)  # ty:ignore[invalid-argument-type]
        return type(self)(array=moved_array, sample_axis=new_sample_axis)

    def __getitem__(self, index: ToIndices) -> NumpyArrayLikeImplementation[D] | D:
        """Get a sample by index.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            The sample at the specified index.
        """
        new_array = cast("Any", self.array)[index]

        if not hasattr(new_array, "ndim"):
            return new_array

        new_sample_axis = track_axis(index, self.sample_axis, self.array.ndim)

        if new_sample_axis is None:
            return new_array

        return type(self)(array=new_array, sample_axis=new_sample_axis)

    def __setitem__(self, index: ToIndices, value: object) -> None:
        """Set a sample by index.

        Args:
            index: The index of the sample to set.
            value: The value to set at the specified index.
        """
        cast("Any", self.array)[index] = value

    def __array__(self, dtype: DTypeLike | None = None, copy: bool | None = None) -> np.ndarray:
        """Get the underlying numpy array.

        Args:
            dtype: Desired data type of the array.
            copy: Whether to return a copy of the array.

        Returns:
            The underlying numpy array.
        """
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

    @override
    def copy(self, order: Order = "C") -> ArraySample[D]:
        """Create a copy of the ArraySample.

        Returns:
            A copy of the ArraySample.
        """
        copied_array = cast("Any", self.array).copy(order=order)
        return type(self)(array=copied_array, sample_axis=self.sample_axis)

    def __eq__(self, value: Any) -> Self:  # noqa: ANN401, PYI032  # ty:ignore[invalid-method-override]
        """Vectorized equality comparison."""
        return np.equal(self, value)

    def __hash__(self) -> int:
        """Compute the hash of the ArraySample."""
        return object.__hash__(self)

    @override
    def __index__(self) -> int:
        return cast("Any", self.array).__index__()

    @override
    def __int__(self) -> int:
        return cast("Any", self.array).__int__()

    @override
    def __bool__(self) -> bool:
        return cast("Any", self.array).__bool__()

    @override
    def __float__(self) -> float:
        return cast("Any", self.array).__float__()

    @override
    def __complex__(self) -> complex:
        return cast("Any", self.array).__complex__()

    @override
    def __iter__(self) -> Iterator[D]:
        """Return an iterator over the first dimension of the underlying array.

        For an iterator over the samples, use the :attr:`samples` property.
        """
        return cast("Any", self.array).__iter__()

    def __array_like__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> ArraySample[D]:
        """Convert to a NumpyArrayLike."""
        if copy:
            return self.copy()

        return self

    def __torch_like__(
        self, dtype: torch.dtype | None = None, /, *, device: torch.device | str | None = None, copy: bool = False
    ) -> TorchTensorLikeImplementation[Any]:
        """Convert to a TorchTensorSample."""
        from probly.representation.torch_like import to_torch_tensor_like  # noqa: PLC0415

        from .torch import TorchTensorSample  # noqa: PLC0415

        tensor = to_torch_tensor_like(self.array, dtype=dtype, device=device, copy=copy)

        return TorchTensorSample(cast("Any", tensor), sample_dim=self.sample_axis)


@array_sample_internals.register(ArraySample)
def _[D: NumpyArrayLike](array: ArraySample[D]) -> ArraySampleInternals[D]:
    """Get the sample dimension of an ArraySample."""
    return ArraySampleInternals(
        create=type(array),
        array=array.array,
        sample_axis=array.sample_axis,
    )


create_sample.register(
    np.number | np.ndarray | float | int | NumpyArrayLikeImplementation,
    ArraySample.from_iterable,
)
