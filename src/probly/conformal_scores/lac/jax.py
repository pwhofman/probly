"""LAC score computation for JAX arrays."""

from __future__ import annotations

import jax.numpy as jnp

from ._common import lac_score


@lac_score.register(jnp.ndarray)
def compute_lac_score_jax(probs: jnp.ndarray, y_cal: jnp.ndarray | None = None) -> jnp.ndarray:
    """Compute the LAC score."""
    probs_jnp = jnp.asarray(probs, dtype=float)
    if probs_jnp.ndim < 1:
        msg = f"probs must have at least one dimension with classes on the last axis, got shape {probs_jnp.shape}."
        raise ValueError(msg)

    scores = 1.0 - probs_jnp
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
