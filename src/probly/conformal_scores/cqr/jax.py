"""JAX implementation for CQR scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from ._common import cqr_score


@cqr_score.register(Array)
def _(y_pred: Array, y_true: Array) -> Array:
    """CQR nonconformity scores for JAX arrays."""
    y = jnp.asarray(y_true, dtype=float)
    pred = jnp.asarray(y_pred, dtype=float)

    if pred.ndim < y.ndim + 1 or pred.shape[-1] != 2:
        msg = (
            "y_pred must have shape (..., 2) or (n_evaluations, ..., 2) matching y_true batch shape; "
            f"got y_pred shape {pred.shape} and y_true shape {y.shape}."
        )
        raise ValueError(msg)
    if pred.ndim == y.ndim + 2:
        pred = pred.mean(axis=0)
    elif pred.ndim != y.ndim + 1:
        msg = (
            "y_pred must match y_true batch rank with a trailing quantile axis, "
            "or include one additional leading evaluation axis."
        )
        raise ValueError(msg)
    lower = pred[..., 0]
    upper = pred[..., 1]

    return jnp.maximum(lower - y, y - upper)
