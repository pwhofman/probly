"""Utils module for probly library."""

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import entropy, intersection_probability
from .switchdispatch import switchdispatch


@entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@intersection_probability.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@entropy.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER))
@intersection_probability.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER))
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "entropy",
    "intersection_probability",
    "switchdispatch",
]
