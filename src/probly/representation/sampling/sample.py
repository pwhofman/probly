"""Classes representing prediction samples."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from lazy_dispatch.singledispatch import lazydispatch
from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR


class Sample[T](ABC):
    """Abstract base class for samples."""

    @abstractmethod
    def __init__(self, samples: list[T]) -> None:
        """Initialize the sample."""
        ...

    def mean(self) -> T:
        """Compute the mean of the sample."""
        msg = "mean method not implemented."
        raise NotImplementedError(msg)

    def std(self, ddof: int = 1) -> T:
        """Compute the standard deviation of the sample."""
        msg = "std method not implemented."
        raise NotImplementedError(msg)

    def var(self, ddof: int = 1) -> T:
        """Compute the variance of the sample."""
        msg = "var method not implemented."
        raise NotImplementedError(msg)


class ListSample[T](list[T], Sample[T]):
    """A sample of predictions stored in a list."""

    def __add__[S](self, other: list[S]) -> ListSample[T | S]:
        """Add two samples together."""
        return type(self)(super().__add__(other))  # type: ignore[operator]


create_sample = lazydispatch[type[Sample], Sample](ListSample, dispatch_on=lambda s: s[0])


@create_sample.register(np.number | np.ndarray | float | int)
class ArraySample[T](Sample[T]):
    """A sample of predictions stored in a numpy array."""

    def __init__(self, samples: list[T]) -> None:
        """Initialize the array sample."""
        self.array: np.ndarray = np.array(samples)

    def mean(self) -> T:
        """Compute the mean of the sample."""
        return self.array.mean(axis=0)  # type: ignore[no-any-return]

    def std(self, ddof: int = 1) -> T:
        """Compute the standard deviation of the sample."""
        return self.array.std(axis=0, ddof=ddof)  # type: ignore[no-any-return]

    def var(self, ddof: int = 1) -> T:
        """Compute the variance of the sample."""
        return self.array.var(axis=0, ddof=ddof)  # type: ignore[no-any-return]


@create_sample.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_sample as torch_sample  # noqa: PLC0414, PLC0415


@create_sample.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_sample as jax_sample  # noqa: PLC0414, PLC0415
