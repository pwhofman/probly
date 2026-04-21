"""JAX implementation for RAPS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp
import jax.random

from ._common import _raps_score_dispatch


@_raps_score_dispatch.register(Array)
def _(
    probs: Array,
    y_cal: Array | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> Array:
    """RAPS Nonconformity-Scores for JAX arrays."""
    probs_jnp = jnp.asarray(probs, dtype=float)
    if probs_jnp.ndim < 1:
        msg = f"probs must have at least one dimension with classes on the last axis, got shape {probs_jnp.shape}."
        raise ValueError(msg)

    n_classes = probs_jnp.shape[-1]

    # sorting indices for descending probabilities
    srt_idx = jnp.argsort(-probs_jnp, axis=-1)
    srt_probs = jnp.take_along_axis(probs_jnp, srt_idx, axis=-1)

    # calculate cumulative sums
    cumsum_probs = jnp.cumsum(srt_probs, axis=-1)

    if randomized:
        u = jax.random.uniform(jax.random.PRNGKey(42), shape=probs_jnp.shape)
        cumsum_probs -= srt_probs * u

    # regularization penalty
    ranks = jnp.arange(1, n_classes + 1, dtype=probs_jnp.dtype).reshape((1,) * (probs_jnp.ndim - 1) + (n_classes,))
    penalty = lambda_reg * jnp.maximum(jnp.array(0.0, dtype=probs_jnp.dtype), ranks - k_reg - 1)
    epsilon_penalty = epsilon * jnp.ones(probs_jnp.shape, dtype=probs_jnp.dtype)

    reg_cumsum = cumsum_probs + penalty + epsilon_penalty

    # sort back to original order
    inv_idx = jnp.argsort(srt_idx, axis=-1)
    scores = jnp.take_along_axis(reg_cumsum, inv_idx, axis=-1)

    if y_cal is not None:
        labels = jnp.asarray(y_cal, dtype=jnp.int32)
        if labels.shape != probs_jnp.shape[:-1]:
            msg = (
                "y_cal must match probs batch shape (all axes except the class axis); "
                f"got y_cal shape {labels.shape} and probs shape {probs_jnp.shape}."
            )
            raise ValueError(msg)
        scores = jnp.take_along_axis(scores, labels[..., None], axis=-1)
        scores = jnp.squeeze(scores, axis=-1)
    return scores
