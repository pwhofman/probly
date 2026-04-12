"""JAX implementation for CQR scores."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ._common import cqr_score_func


@cqr_score_func.register(Array)
def _(y_pred: Array, y_true: Array) -> Array:
    """CQR nonconformity scores for JAX arrays."""
    y_true = jnp.ravel(y_true)

    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    lower = y_pred[:, 0]
    upper = y_pred[:, 1]

    return jnp.maximum(lower - y_true, y_true - upper)
