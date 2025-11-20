"""Classes representing prediction samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, TypedDict, Unpack, override

import numpy as np

from lazy_dispatch.singledispatch import lazydispatch
from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR
from probly.representation.sampling.sample_axis_tracking import Index, track_axis

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import DTypeLike

type SampleDim = int | Literal["auto"]


class SampleParams(TypedDict, total=False):
    """Default parameters for sample creation."""

    sample_dim: SampleDim


class SampleFactory[T, S: Sample](Protocol):
    """Protocol for the creation of samples."""

    def __call__(self, samples: Iterable[T], **kwargs: Unpack[SampleParams]) -> S:
        """Create a sample from the given predictions.

        Args:
            samples: The predictions to create the sample from.
            kwargs: Parameters for sample creation.

        Returns:
            The created sample.
        """


class Sample[T](ABC):
    """Abstract base class for samples."""

    @classmethod
    @abstractmethod
    def from_iterable(cls, samples: Iterable[T], **kwargs: Unpack[SampleParams]) -> Self:
        """Create an Sample from an iterable of samples.

        Args:
            samples: The predictions to create the sample from.
            kwargs: Parameters for sample creation.

        Returns:
            The created ArraySample.
        """
        ...

    @classmethod
    def from_sample(cls, sample: Sample[T], **kwargs: Unpack[SampleParams]) -> Self:
        """Create a new Sample from an existing Sample.

        Args:
            sample: The sample to create the new sample from.
            kwargs: Parameters for sample creation.

        Returns:
            The created Sample.
        """
        return cls.from_iterable(list(sample.samples), **kwargs)

    @property
    @abstractmethod
    def samples(self) -> Iterable[T]:
        """Return an iterator over the samples."""
        ...

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return sum(1 for _ in self.samples)

    def concat(self, other: Sample[T]) -> Self:
        """Append another sample to this sample."""
        return type(self).from_iterable(samples=(sample for s in (self, other) for sample in s.samples))

    def sample_mean(self) -> T:
        """Compute the mean of the sample."""
        msg = "mean method not implemented."
        raise NotImplementedError(msg)

    def sample_std(self, ddof: int = 1) -> T:
        """Compute the standard deviation of the sample."""
        msg = "std method not implemented."
        raise NotImplementedError(msg)

    def sample_var(self, ddof: int = 1) -> T:
        """Compute the variance of the sample."""
        msg = "var method not implemented."
        raise NotImplementedError(msg)


class ListSample[T](list[T], Sample[T]):
    """A sample of predictions stored in a list."""

    @classmethod
    def from_iterable(cls, samples: Iterable[T], sample_dim: SampleDim = "auto") -> Self:
        """Create a ListSample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            sample_dim: The dimension along which samples are organized.

        Returns:
            The created ListSample.
        """
        if sample_dim != "auto":
            msg = "List-based samples do not support a user-defined sample_dim."
            raise ValueError(msg)

        return cls(samples)

    @property
    def samples(self) -> Sequence[T]:
        """Return an iterator over the samples."""
        return self

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return len(self)

    def concat(self, other: Sample[T]) -> Self:
        """Creates a new sample by concatenating another sample to this sample."""
        return type(self)(self + list(other.samples))


create_sample = lazydispatch[SampleFactory, Sample](ListSample.from_iterable, dispatch_on=lambda s: s[0])

type Numeric = np.number | np.ndarray | float | int

STRUCTURE_PRESERVING_NP_FUNCTIONS = {
    np.argmax,
    np.argmin,
    np.argsort,
}


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArraySample[T: Numeric](Sample[T], np.lib.mixins.NDArrayOperatorsMixin):
    """A sample of predictions stored in a numpy array."""

    array: np.ndarray
    sample_dim: int

    def __post_init__(self) -> None:
        """Validate the sample_dim."""
        if self.sample_dim >= self.array.ndim:
            msg = f"sample_dim {self.sample_dim} out of bounds for array with ndim {self.array.ndim}."
            raise ValueError(msg)
        if self.sample_dim < 0:
            super().__setattr__("sample_dim", self.array.ndim + self.sample_dim)

    @classmethod
    def from_iterable(cls, samples: Iterable[T], sample_dim: SampleDim = "auto", dtype: DTypeLike = None) -> Self:
        """Create an ArraySample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            sample_dim: The dimension along which samples are organized.
            dtype: Desired data type of the array.

        Returns:
            The created ArraySample.
        """
        if isinstance(samples, np.ndarray):
            if sample_dim == "auto":
                if samples.ndim == 0:
                    msg = "Cannot infer sample_dim for 0-dimensional array."
                    raise ValueError(msg)
                sample_dim = 0 if samples.ndim == 1 else 1
            if sample_dim != 0:
                samples = np.moveaxis(samples, 0, sample_dim)
            if dtype is not None:
                samples = samples.astype(dtype)
        else:
            if not isinstance(samples, Sequence):
                samples = list(samples)
            if sample_dim == "auto":
                if len(samples) == 0:
                    msg = "Cannot infer sample_dim for empty samples."
                    raise ValueError(msg)
                first_sample = samples[0]
                sample_dim = (0 if first_sample.ndim == 0 else 1) if isinstance(first_sample, np.ndarray) else 0
            samples = np.stack(samples, axis=sample_dim, dtype=dtype)

        return cls(array=samples, sample_dim=sample_dim)

    def sample_mean(self) -> T:
        """Compute the mean of the sample."""
        return self.array.mean(axis=self.sample_dim)  # type: ignore[no-any-return]

    def sample_std(self, ddof: int = 1) -> T:
        """Compute the standard deviation of the sample."""
        return self.array.std(axis=self.sample_dim, ddof=ddof)  # type: ignore[no-any-return]

    def sample_var(self, ddof: int = 1) -> T:
        """Compute the variance of the sample."""
        return self.array.var(axis=self.sample_dim, ddof=ddof)  # type: ignore[no-any-return]

    @override
    @classmethod
    def from_sample(cls, sample: Sample[T], sample_dim: SampleDim = "auto", dtype: DTypeLike = None) -> Self:
        if isinstance(sample, ArraySample):
            sample_array = sample.array

            if dtype is not None:
                sample_array = sample_array.astype(dtype)

            in_sample_dim = sample.sample_dim
            if sample_dim not in ("auto", in_sample_dim):
                sample_array = np.moveaxis(sample_array, in_sample_dim, sample_dim)  # type: ignore[arg-type]
                in_sample_dim = sample_dim  # type: ignore[assignment]
            return cls(array=sample_array, sample_dim=in_sample_dim)

        return cls.from_iterable(sample.samples, sample_dim=sample_dim, dtype=dtype)

    def __len__(self) -> int:
        """Return the len of the array."""
        return len(self.array)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the underlying array."""
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying array."""
        return self.array.shape  # type: ignore[no-any-return]

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return self.array.shape[self.sample_dim]  # type: ignore[no-any-return]

    @property
    def samples(self) -> np.ndarray:
        """Return an iterator over the samples."""
        if self.sample_dim == 0:
            return self.array
        return np.moveaxis(self.array, self.sample_dim, 0)

    @override
    def concat(self, other: Sample[T]) -> Self:
        if isinstance(other, ArraySample):
            other_array = np.moveaxis(other.array, other.sample_dim, self.sample_dim)
        else:
            other_array = np.stack(list(other.samples), axis=self.sample_dim, dtype=self.array.dtype)

        concatenated = np.concatenate((self.array, other_array), axis=self.sample_dim)

        return type(self)(array=concatenated, sample_dim=self.sample_dim)

    def move_sample_dim(self, new_sample_dim: int) -> ArraySample[T]:
        """Return a new ArraySample with the sample dimension moved to new_sample_dim.

        Args:
            new_sample_dim: The new sample dimension.

        Returns:
            A new ArraySample with the sample dimension moved.
        """
        moved_array = np.moveaxis(self.array, self.sample_dim, new_sample_dim)
        return type(self)(array=moved_array, sample_dim=new_sample_dim)

    def __getitem__(self, index: Index) -> Self | T | np.ndarray:
        """Get a sample by index.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            The sample at the specified index.
        """
        new_array = self.array[index]
        new_sample_dim = track_axis(index, self.sample_dim, self.array.ndim)

        if not isinstance(new_array, np.ndarray) or new_sample_dim is None:
            return new_array

        return type(self)(array=new_array, sample_dim=new_sample_dim)

    def __setitem__(self, index: int | slice | np.ndarray, value: T | np.ndarray) -> None:
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

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:  # noqa: ANN401, C901, PLR0912
        """Handle numpy ufuncs.

        Args:
            ufunc: The ufunc to apply.
            method: The method of the ufunc.
            inputs: The input arrays.
            kwargs: Additional keyword arguments.

        Returns:
            The result of applying the ufunc.
        """
        new_sample_dim = self.sample_dim
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
            axis: int | tuple[int] = kwargs.get("axis", 0)
            axes: tuple[int] = axis if isinstance(axis, tuple) else (axis,)
            keepdims: bool = kwargs.get("keepdims", False)
            if (method != "reduce" or not keepdims) and self.sample_dim in axes:
                return result
            if not keepdims:
                for a in axes:
                    if a < self.sample_dim:
                        new_sample_dim -= 1
        elif method in ("at", "outer"):
            return result

        if outs is not None:
            if len(outs) == 1:
                return outs[0]
            return outs

        if isinstance(result, np.ndarray):
            return type(self)(result, sample_dim=new_sample_dim)
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
        return func._implementation(*args, **kwargs)  # type: ignore[attr-defined]  # noqa: SLF001

    def copy(self) -> ArraySample[T]:
        """Create a copy of the ArraySample.

        Returns:
            A copy of the ArraySample.
        """
        return type(self)(array=self.array.copy(), sample_dim=self.sample_dim)


create_sample.register(np.number | np.ndarray | float | int, ArraySample.from_iterable)


@create_sample.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_sample as torch_sample  # noqa: PLC0414, PLC0415


@create_sample.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_sample as jax_sample  # noqa: PLC0414, PLC0415
