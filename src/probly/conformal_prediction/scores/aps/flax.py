"""Flax/JAX implementation for APS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from .common import register


def aps_score_jax(probs: Array) -> Array:
    """Compute APS scores for JAX arrays (keeping data on GPU/TPU)."""
    # sort indices in descending order
    srt_idx = jnp.argsort(-probs, axis=1)

    # sorted probabilities
    srt_probs = jnp.take_along_axis(probs, srt_idx, axis=1)

    # calculate cumulative sums
    cumsum_probs = jnp.cumsum(srt_probs, axis=1)

    # sort back to original positions without in-place writes
    inv_idx = jnp.argsort(srt_idx, axis=1)
    return jnp.take_along_axis(cumsum_probs, inv_idx, axis=1)


register(Array, aps_score_jax)
register("jaxlib.xla_extension.ArrayImpl", aps_score_jax)  # for compatibility with different JAX versions
