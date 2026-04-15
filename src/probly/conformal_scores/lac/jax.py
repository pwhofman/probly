"""LAC score computation for JAX arrays."""

from __future__ import annotations

import jax.numpy as jnp

from ._common import lac_score_func


@lac_score_func.register(jnp.ndarray)
def compute_lac_score_jax(probs: jnp.ndarray, y_cal: jnp.ndarray | None = None) -> jnp.ndarray:
    """Compute the LAC score."""
    scores = 1.0 - probs
    if y_cal is not None:
        scores = scores[jnp.arange(len(probs)), y_cal]
    return scores
