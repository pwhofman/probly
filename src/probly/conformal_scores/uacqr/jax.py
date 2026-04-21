"""JAX implementation for UACQR scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from ._common import uacqr_score


@uacqr_score.register(Array)
def _(y_pred: Array, y_true: Array) -> Array:
    """UACQR nonconformity scores for JAX arrays."""
    y = jnp.asarray(y_true, dtype=float)
    pred = jnp.asarray(y_pred, dtype=float)

    if pred.ndim != y.ndim + 2 or pred.shape[-1] != 2:
        msg = (
            "intervals must have shape (n_estimations, ..., 2) with batch shape matching y_true; "
            f"got y_pred shape {pred.shape} and y_true shape {y.shape}."
        )
        raise ValueError(msg)

    std = jnp.std(pred, axis=0)
    mean_intervals = jnp.mean(pred, axis=0)

    lower = mean_intervals[..., 0]
    upper = mean_intervals[..., 1]
    std_lo = std[..., 0]
    std_hi = std[..., 1]

    return jnp.maximum((lower - y) / std_lo, (y - upper) / std_hi)
