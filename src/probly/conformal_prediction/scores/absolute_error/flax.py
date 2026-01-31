"""Absolute Error Score implementation."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from .common import register


def absolute_error_jax(y_true: Array, y_pred: Array) -> Array:
    """Compute absolute error |y - y_hat|."""
    return jnp.abs(y_true - y_pred)


register(Array, absolute_error_jax)
register("jaxlib.xla_extension.ArrayImpl", absolute_error_jax)
