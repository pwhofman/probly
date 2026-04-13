"""JAX implementation for SAPS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp
import jax.random

from ._common import saps_score_func


@saps_score_func.register(Array)
def _(
    probs: Array,
    y_cal: Array | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> Array:
    """SAPS Nonconformity-Scores for JAX arrays."""
    probs = jnp.asarray(probs, dtype=float)
    n_samples, n_classes = probs.shape

    if randomized:
        u = jax.random.uniform(jax.random.PRNGKey(42), shape=(n_samples, n_classes))
    else:
        u = jnp.zeros((n_samples, n_classes), dtype=probs.dtype)

    max_probs = jnp.max(probs, axis=1, keepdims=True)
    sort_idx = jnp.argsort(-probs, axis=1)
    ranks_zero_based = jnp.argsort(sort_idx, axis=1)
    ranks = ranks_zero_based + 1

    scores = jnp.where(ranks == 1, u * max_probs, max_probs + (ranks - 2 + u) * lambda_val)

    if y_cal is not None:
        scores = scores[jnp.arange(n_samples), y_cal]
    return jnp.asarray(scores, dtype=float)
