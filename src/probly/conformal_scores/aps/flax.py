"""Flax/JAX implementation for APS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp
import jax.random

from ._common import _aps_score_dispatch


@_aps_score_dispatch.register(Array)
def _(probs: Array, y_cal: Array | None = None, randomized: bool = True) -> Array:
    """APS Nonconformity-Scores for JAX arrays."""
    probs_jnp = jnp.asarray(probs, dtype=float)

    if probs_jnp.ndim < 1:
        msg = f"probs must have at least one dimension with classes on the last axis, got shape {probs_jnp.shape}."
        raise ValueError(msg)

    # sorting indices for descending probabilities
    srt_idx = jnp.argsort(-probs_jnp, axis=-1)

    # get sorted probabilities
    srt_probs = jnp.take_along_axis(probs_jnp, srt_idx, axis=-1)

    # calculate cumulative sums
    cumsum_probs = jnp.cumsum(srt_probs, axis=-1)

    # sort back to original positions without in-place writes
    inv_idx = jnp.argsort(srt_idx, axis=-1)

    if randomized:
        u = jax.random.uniform(jax.random.PRNGKey(42), shape=probs_jnp.shape)
        cumsum_probs -= srt_probs * u

    scores = jnp.take_along_axis(cumsum_probs, inv_idx, axis=-1)
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
