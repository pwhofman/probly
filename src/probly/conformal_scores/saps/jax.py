"""JAX implementation for SAPS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp
import jax.random

from ._common import _saps_score_dispatch


@_saps_score_dispatch.register(Array)
def _(
    probs: Array,
    y_cal: Array | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> Array:
    """SAPS Nonconformity-Scores for JAX arrays."""
    probs_jnp = jnp.asarray(probs, dtype=float)
    if probs_jnp.ndim < 1:
        msg = f"probs must have at least one dimension with classes on the last axis, got shape {probs_jnp.shape}."
        raise ValueError(msg)

    if randomized:
        u = jax.random.uniform(jax.random.PRNGKey(42), shape=probs_jnp.shape)
    else:
        u = jnp.zeros(probs_jnp.shape, dtype=probs_jnp.dtype)

    max_probs = jnp.max(probs_jnp, axis=-1, keepdims=True)
    sort_idx = jnp.argsort(-probs_jnp, axis=-1)
    ranks_zero_based = jnp.argsort(sort_idx, axis=-1)
    ranks = ranks_zero_based + 1

    scores = jnp.where(ranks == 1, u * max_probs, max_probs + (ranks - 2 + u) * lambda_val)

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
    return jnp.asarray(scores, dtype=float)
