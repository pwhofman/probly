"""JAX implementation for CQR-r scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from ._common import _EPS, cqr_r_score_func, weight_func


@cqr_r_score_func.register(Array)
def _(y_pred: Array, y_true: Array) -> Array:
    """CQR-r nonconformity scores for JAX arrays."""
    y_true = jnp.ravel(y_true)

    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    lower = y_pred[:, 0]
    upper = y_pred[:, 1]
    width = jnp.maximum(upper - lower, _EPS)

    return jnp.maximum(lower - y_true, y_true - upper) / width


@weight_func.register(Array)
def _(y_pred: Array) -> tuple[Array, Array]:
    """Weights for CQR-r score normalization for JAX arrays."""
    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    width = y_pred[:, 1] - y_pred[:, 0]
    weight = 1 / jnp.maximum(width, _EPS)
    return weight, weight
