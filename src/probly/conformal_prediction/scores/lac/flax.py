"""Flax/JAX implementation for LAC scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from .common import register


def lac_score_jax(probs: Array) -> Array:
    """Compute LAC scores for JAX arrays."""
    lac_scores = 1.0 - probs
    return jnp.asarray(lac_scores)


register(Array, lac_score_jax)
register("jaxlib.xla_extension.ArrayImpl", lac_score_jax)
