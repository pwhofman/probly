"""JAX implementation for absolute error scores."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ._common import absolute_error_score_func


@absolute_error_score_func.register(Array)
def _(y_true: Array, y_pred: Array) -> Array:
    """Absolute error for JAX arrays."""
    return jnp.abs(y_true - y_pred)
