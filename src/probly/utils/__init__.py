"""Utils module for probly library."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import entropy, intersection_probability


@entropy.delayed_register(TORCH_VALUE)
@intersection_probability.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@entropy.delayed_register(JAX_VALUE)
@intersection_probability.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "entropy",
    "intersection_probability",
]
