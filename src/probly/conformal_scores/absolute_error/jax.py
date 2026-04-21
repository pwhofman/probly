"""JAX implementation for absolute error scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from ._common import absolute_error_score


@absolute_error_score.register(Array)
def _(y_pred: Array, y_true: Array) -> Array:
    """Absolute error for JAX arrays."""
    y_pred_j = jnp.asarray(y_pred, dtype=float)
    y_true_j = jnp.asarray(y_true, dtype=float)

    if y_pred_j.ndim == y_true_j.ndim + 1:
        y_pred_j = y_pred_j.mean(axis=0)
    elif y_pred_j.ndim != y_true_j.ndim:
        msg = (
            "y_pred must match y_true shape or add a leading evaluation axis; "
            f"got y_pred shape {y_pred_j.shape} and y_true shape {y_true_j.shape}."
        )
        raise ValueError(msg)

    return jnp.abs(y_true_j - y_pred_j)
