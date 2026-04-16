"""JAX implementation for absolute error scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from ._common import absolute_error_score_func


@absolute_error_score_func.register(Array)
def _(y_pred: Array, y_true: Array) -> Array:
    """Absolute error for JAX arrays."""
    if y_pred.ndim > 2:
        msg = (
            "y_pred must have shape (n_evaluations, n_samples) or (n_samples,), "
            f"got {y_pred.shape}. The n_evaluations dimension is optional and "
            "will be averaged over if present."
        )
        raise ValueError(msg)
    if y_pred.ndim == 2:
        y_pred = y_pred.mean(axis=0)
    return jnp.abs(y_true - y_pred)
