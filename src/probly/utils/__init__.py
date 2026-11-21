"""Utils module for probly library."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from . import numpy as numpy
from ._common import elementwise_entropy, entropy, intersection_probability


@elementwise_entropy.delayed_register(TORCH_VALUE)
@entropy.delayed_register(TORCH_VALUE)
@intersection_probability.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@elementwise_entropy.delayed_register(JAX_VALUE)
@entropy.delayed_register(JAX_VALUE)
@intersection_probability.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "elementwise_entropy",
    "entropy",
    "intersection_probability",
]
