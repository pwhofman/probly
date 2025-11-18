"""Classes representing credal sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_dispatch.singledispatch import lazydispatch
from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR
from probly.representation.sampling.sample import ArraySample

if TYPE_CHECKING:
    import numpy as np


class CredalSet[T]:
    """Base class for credal sets."""

    def lower(self) -> T:
        """Compute the lower envelope of the credal set."""
        msg = "lower method not implemented."
        raise NotImplementedError(msg)

    def upper(self) -> T:
        """Compute the upper envelope of the credal set."""
        msg = "upper method not implemented."
        raise NotImplementedError(msg)


credal_set_from_sample = lazydispatch[type[CredalSet], CredalSet](CredalSet)


@credal_set_from_sample.register(ArraySample)
class ArrayCredalSet[T](CredalSet[T]):
    """A credal set of predictions stored in a numpy array."""

    def __init__(self, sample: ArraySample) -> None:
        """Initialize the array credal set."""
        self.array: np.ndarray = sample.array

    def lower(self) -> T:
        """Compute the lower envelope of the credal set."""
        return self.array.min(axis=1)  # type: ignore[no-any-return]

    def upper(self) -> T:
        """Compute the upper envelope of the credal set."""
        return self.array.max(axis=1)  # type: ignore[no-any-return]


@credal_set_from_sample.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0414, PLC0415


@credal_set_from_sample.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0414, PLC0415
