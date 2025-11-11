"""Classes representing prediction samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, TypedDict, Unpack

import numpy as np

from lazy_dispatch.singledispatch import lazydispatch
from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR

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

    @abstractmethod
    def concat(self, other: Self) -> Self:
        """Append another sample to this sample."""
        ...

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


create_sample = lazydispatch[type[Sample], Sample](ListSample, dispatch_on=lambda s: s[0])

Numeric = np.number | np.ndarray | float | int


@create_sample.register(Numeric)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArraySample[T: Numeric](Sample[T], np.lib.mixins.NDArrayOperatorsMixin):
    """A sample of predictions stored in a numpy array."""

    array: np.ndarray
    sample_dim: int

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
            samples = np.stack(samples, axis=sample_dim)

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

    @property
    def samples(self) -> np.ndarray:
        """Return an iterator over the samples."""
        if self.sample_dim == 0:
            return self.array
        return np.moveaxis(self.array, self.sample_dim, 0)

    def __len__(self) -> int:
        """Return the len of the array."""
        return len(self.array)

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return self.array.shape[self.sample_dim]  # type: ignore[no-any-return]

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

    def __repr__(self) -> str:
        """Return the string representation of the sample."""
        return f"ArraySample(size={len(self)}, {self.array!r})"

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> ArraySample:  # noqa: ANN401
        """Handle numpy ufuncs.

        Args:
            ufunc: The ufunc to apply.
            method: The method of the ufunc.
            inputs: The input arrays.
            kwargs: Additional keyword arguments.

        Returns:
            The result of applying the ufunc.
        """
        arrays = [x.array if isinstance(x, ArraySample) else x for x in inputs]
        result = getattr(ufunc, method)(*arrays, **kwargs)
        if isinstance(result, np.ndarray):
            return type(self)(list(result), sample_dim=self.sample_dim)
        return result


@create_sample.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_sample as torch_sample  # noqa: PLC0414, PLC0415


@create_sample.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_sample as jax_sample  # noqa: PLC0414, PLC0415
