"""Uncertainty measures for credal sets."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from . import numpy as numpy
from ._common import (
    _constant_like,
    _take_along_axis,
    generalized_hartley,
    lower_entropy,
    upper_entropy,
)


@_constant_like.delayed_register(TORCH_VALUE)
@_take_along_axis.delayed_register(TORCH_VALUE)
@upper_entropy.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@_constant_like.delayed_register(JAX_VALUE)
@_take_along_axis.delayed_register(JAX_VALUE)
@upper_entropy.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "generalized_hartley",
    "lower_entropy",
    "upper_entropy",
]
