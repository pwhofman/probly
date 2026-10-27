"""This module contains the quantile-based conformal prediction methods."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import calculate_quantile, calculate_weighted_quantile


@calculate_quantile.delayed_register(TORCH_VALUE)
@calculate_weighted_quantile.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@calculate_quantile.delayed_register(JAX_VALUE)
@calculate_weighted_quantile.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415
