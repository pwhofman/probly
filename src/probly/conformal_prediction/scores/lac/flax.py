"""Flax/JAX implementation for LAC scores."""

from __future__ import annotations

from jax import Array

from .common import register


def lac_score_jax(probs: Array) -> Array:
    """Compute LAC scores for JAX arrays."""
    lac_scores = 1.0 - probs
    return lac_scores  # shape: (n_samples, n_classes)


register(Array, lac_score_jax)
register("jaxlib.xla_extension.ArrayImpl", lac_score_jax)
