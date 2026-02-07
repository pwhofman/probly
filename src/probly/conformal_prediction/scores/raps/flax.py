"""Flax/JAX implementation for RAPS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from .common import register


def raps_score_jax(
    probs: Array,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> Array:
    """Compute RAPS scores for JAX arrays.

    For each sample, classes are sorted by descending probability. The score for each
    class is the cumulative sum up to that class in the sorted order, plus a rank-based
    regularization penalty and a small epsilon term.

    Returned shape: (n_samples, n_classes)
    """
    n_samples, n_classes = probs.shape

    # sort class indices by descending probability (per sample)
    srt_idx = jnp.argsort(-probs, axis=1)

    # reorder probabilities into sorted order
    srt_probs = jnp.take_along_axis(probs, srt_idx, axis=1)

    # cumulative sums along sorted order
    cumsum_probs = jnp.cumsum(srt_probs, axis=1)

    # rank-based regularization in sorted space:
    # penalty depends on the rank (1..K) and starts after the first k_reg+1 entries.
    ranks = jnp.arange(1, n_classes + 1, dtype=probs.dtype).reshape(1, -1)  # (1, K)
    penalty = lambda_reg * jnp.maximum(jnp.array(0.0, dtype=probs.dtype), ranks - k_reg - 1)

    # small constant added to all entries
    epsilon_penalty = epsilon * jnp.ones((n_samples, n_classes), dtype=probs.dtype)

    # combine components in sorted space
    reg_cumsum = cumsum_probs + penalty + epsilon_penalty

    # map back to original class order (no in-place writes)
    inv_idx = jnp.argsort(srt_idx, axis=1)
    return jnp.take_along_axis(reg_cumsum, inv_idx, axis=1)


register(Array, raps_score_jax)
register("jaxlib.xla_extension.ArrayImpl", raps_score_jax)  # compatibility with different JAX versions
