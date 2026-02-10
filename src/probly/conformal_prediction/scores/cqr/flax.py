"""Flax/JAX implementation for CQR scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from .common import register


def cqr_score_jax(y_true: Array, y_pred: Array) -> Array:
    """Compute CQR nonconformity scores for JAX arrays.

    Parameters
    ----------
    y_true:
        True targets as JAX array of shape ``(n_samples,)``.
    y_pred:
        Predicted lower and upper quantiles as JAX array of shape
        ``(n_samples, 2)``.

    Returns:
    -------
    Array
        One-dimensional array of nonconformity scores with shape
        ``(n_samples,)``.
    """
    y_true = jnp.ravel(y_true)
    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    lower = y_pred[:, 0]
    upper = y_pred[:, 1]

    diff_lower = lower - y_true
    diff_upper = y_true - upper

    scores = jnp.maximum(diff_lower, diff_upper)
    return scores


register(Array, cqr_score_jax)
register("jaxlib.xla_extension.ArrayImpl", cqr_score_jax)  # for older JAX backends
