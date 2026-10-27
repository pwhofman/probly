"""Uncertainty measures for credal sets."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import (
    generalized_hartley,
    lower_entropy,
    upper_entropy,
)


@upper_entropy.delayed_register(TORCH_VALUE)
@lower_entropy.delayed_register(TORCH_VALUE)
@generalized_hartley.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@upper_entropy.delayed_register(JAX_VALUE)
@lower_entropy.delayed_register(JAX_VALUE)
@generalized_hartley.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "generalized_hartley",
    "lower_entropy",
    "upper_entropy",
]
