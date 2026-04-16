"""JAX implementation for CQR scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from ._common import cqr_score_func


@cqr_score_func.register(Array)
def _(y_pred: Array, y_true: Array) -> Array:
    """CQR nonconformity scores for JAX arrays."""
    y_true = jnp.ravel(y_true)

    if y_pred.ndim > 3 or y_pred.shape[-1] != 2:
        msg = (
            "y_pred must have shape (n_evaluations, n_samples, 2), "
            f"got {y_pred.shape}. The n_evaluations dimension is optional and "
            "will be averaged over if present."
        )
        raise ValueError(msg)
    if y_pred.ndim == 3:
        y_pred = y_pred.mean(axis=0)
    lower = y_pred[:, 0]
    upper = y_pred[:, 1]

    return jnp.maximum(lower - y_true, y_true - upper)
