"""JAX implementation for RAPS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp
import jax.random

from ._common import raps_score_func


@raps_score_func.register(Array)
def _(
    probs: Array,
    y_cal: Array | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> Array:
    """RAPS Nonconformity-Scores for JAX arrays."""
    n_samples, n_classes = probs.shape

    # sorting indices for descending probabilities
    srt_idx = jnp.argsort(-probs, axis=1)
    srt_probs = jnp.take_along_axis(probs, srt_idx, axis=1)

    # calculate cumulative sums
    cumsum_probs = jnp.cumsum(srt_probs, axis=1)

    if randomized:
        u = jax.random.uniform(jax.random.PRNGKey(42), shape=probs.shape)
        cumsum_probs -= srt_probs * u

    # regularization penalty
    ranks = jnp.arange(1, n_classes + 1, dtype=probs.dtype).reshape(1, -1)
    penalty = lambda_reg * jnp.maximum(jnp.array(0.0, dtype=probs.dtype), ranks - k_reg - 1)
    epsilon_penalty = epsilon * jnp.ones((n_samples, n_classes), dtype=probs.dtype)

    reg_cumsum = cumsum_probs + penalty + epsilon_penalty

    # sort back to original order
    inv_idx = jnp.argsort(srt_idx, axis=1)
    scores = jnp.take_along_axis(reg_cumsum, inv_idx, axis=1)

    if y_cal is not None:
        scores = scores[jnp.arange(n_samples), y_cal]
    return scores
