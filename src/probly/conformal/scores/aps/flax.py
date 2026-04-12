"""Flax/JAX implementation for APS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from ._common import aps_score_func

@aps_score_func.register(Array)
def _(probs: Array, y_cal: Array | None = None, randomized: bool = True) -> Array:
    """APS Nonconformity-Scores for JAX arrays."""
    probs_jnp = jnp.asarray(probs)

    # sorting indices for descending probabilities
    srt_idx = jnp.argsort(-probs_jnp, axis=1)

    # get sorted probabilities
    srt_probs = jnp.take_along_axis(probs_jnp, srt_idx, axis=1)

    # calculate cumulative sums
    cumsum_probs = jnp.cumsum(srt_probs, axis=1)

    # sort back to original positions without in-place writes
    inv_idx = jnp.argsort(srt_idx, axis=1)

    if randomized:
        U = jnp.random.uniform(low=0, high=1, size=probs_jnp.shape)
        cumsum_probs -= srt_probs * U

    scores = jnp.take_along_axis(cumsum_probs, inv_idx, axis=1)
    if y_cal is not None:
        relevant_indices = jnp.arange(probs_jnp.shape[0]), y_cal
        scores = scores[relevant_indices]
    return scores