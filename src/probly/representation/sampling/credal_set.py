"""Classes representing credal sets."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from lazy_dispatch.singledispatch import lazydispatch
from probly.lazy_types import TORCH_TENSOR


class CredalSet[T](ABC):
    """Abstract base class for credal sets."""

    @abstractmethod
    def __init__(self, samples: list[T]) -> None:
        """Initialize the credal set."""
        ...

    def lower(self) -> T:
        """Compute the lower envelope of the credal set."""
        msg = "lower method not implemented."
        raise NotImplementedError(msg)

    def upper(self) -> T:
        """Compute the upper envelope of the credal set."""
        msg = "upper method not implemented."
        raise NotImplementedError(msg)


class ListCredalSet[T](list[T], CredalSet[T]):
    """A credal set of predictions stored in a list."""

    def __add__[S](self, other: list[S]) -> ListCredalSet[T | S]:
        """Add two credal sets together by taking the union."""
        return type(self)(set(self) | set(other))  # type: ignore[operator]


create_credal_set = lazydispatch[type[CredalSet], CredalSet](ListCredalSet, dispatch_on=lambda s: s[0])


@create_credal_set.register(np.number | np.ndarray | float | int)
class ArrayCredalSet[T](CredalSet[T]):
    """A credal set of predictions stored in a numpy array."""

    def __init__(self, samples: list[T]) -> None:
        """Initialize the array credal set."""
        self.array: np.ndarray = np.array(samples).transpose(1, 0, 2)  # we use [instances, samples, classes]

    def lower(self) -> T:
        """Compute the lower envelope of the credal set."""
        return self.array.min(axis=1)  # type: ignore[no-any-return]

    def upper(self) -> T:
        """Compute the upper envelope of the credal set."""
        return self.array.max(axis=1)  # type: ignore[no-any-return]


@create_credal_set.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_credal_set as torch_credal_set  # noqa: PLC0414, PLC0415
