"""Flax/JAX implementation for SAPS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from .common import register


def saps_score_jax(
    probs: Array,
    lambda_val: float,
    u: Array,
) -> Array:
    """Compute SAPS Nonconformity Score for JAX arrays.

    Args:
        probs: 1D array with softmax probabilities.
        lambda_val: Lambda value for SAPS.
        u: Optional random value in [0,1). If None, generated from key.
        key: JAX random key for generating u if not provided.

    Returns:
        float: SAPS nonconformity score.
    """
    # convert to jax arrays
    u = jnp.asarray(u, dtype=probs.dtype)

    # get max probabilities for each sample
    max_probs = jnp.max(probs, axis=1, keepdims=True)

    # get ranks for each label, argsort along axis=1 in descending order
    sort_idx = jnp.argsort(-probs, axis=1)

    # find the rank (1-based) of each label
    # compare each position in sorted_indices with the corresponding label
    ranks_zero_based = jnp.argsort(sort_idx, axis=1)
    ranks = ranks_zero_based + 1  # +1 for 1-based rank

    term_rank1 = u * max_probs
    term_rank_other = max_probs + (ranks - 2 + u) * lambda_val

    scores = jnp.where(ranks == 1, term_rank1, term_rank_other)

    return scores


# register the implementation
register(Array, saps_score_jax)
register("jaxlib.xla_extension.ArrayImpl", saps_score_jax)
