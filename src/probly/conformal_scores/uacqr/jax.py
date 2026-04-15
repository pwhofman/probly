"""JAX implementation for UACQR scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from ._common import uacqr_score_func


@uacqr_score_func.register(Array)
def _(y_pred: Array, y_true: Array) -> Array:
    """UACQR nonconformity scores for JAX arrays."""
    y_true = jnp.ravel(y_true)

    if y_pred.ndim != 3 or y_pred.shape[2] != 2:
        msg = f"intervals must have shape (n_estimations, n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    std = jnp.std(y_pred, axis=0)
    mean_intervals = jnp.mean(y_pred, axis=0)

    lower = mean_intervals[:, 0]
    upper = mean_intervals[:, 1]
    std_lo = std[:, 0]
    std_hi = std[:, 1]

    return jnp.maximum((lower - y_true) / std_lo, (y_true - upper) / std_hi)
