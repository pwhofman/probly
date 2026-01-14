"""Flax/JAX implementation for SAPS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from .common import register


class InvalidProbsDimensionError(ValueError):
    """Raised when the probs array has invalid dimensions."""


class InvalidLabelError(ValueError):
    """Raised when the label is invalid."""


def saps_score_jax(
    probs: Array,
    label: int,
    lambda_val: float = 0.1,
    u: float | None = None,
    key: jrandom.PRNGKeyArray | None = None,
) -> float:
    """Compute SAPS Nonconformity Score for JAX arrays.

    Args:
        probs: 1D array with softmax probabilities.
        label: True label index.
        lambda_val: Lambda value for SAPS.
        u: Optional random value in [0,1). If None, generated from key.
        key: JAX random key for generating u if not provided.

    Returns:
        float: SAPS nonconformity score.
    """
    if probs.ndim == 2:
        if probs.shape[0] != 1:
            raise InvalidProbsDimensionError
        probs = probs[0]

    if probs.ndim != 1:
        raise InvalidProbsDimensionError

    if not (0 <= label < probs.shape[0]):
        raise InvalidLabelError

    if u is None:
        if key is None:
            u = float(np.random.default_rng().uniform())
        else:
            key, subkey = jrandom.split(key)
            u = float(jrandom.uniform(subkey, shape=()).item())

    max_prob = float(jnp.max(probs))

    sorted_indices = jnp.argsort(-probs)

    # find pos of label
    matches = sorted_indices == label
    pos = jnp.nonzero(matches, size=1)[0]

    if pos.size == 0:
        raise ValueError

    # convert to 1-based rank
    rank = int(pos[0].item()) + 1

    if rank == 1:
        return float(u * max_prob)
    # For any rank > 1, SAPS uses max_prob + lambda * (1 + u)
    return float(max_prob + lambda_val * (1 + u))


# Optional batch helper function for JAX
def saps_score_jax_batch(
    probs: Array,
    labels: Array,
    lambda_val: float = 0.1,
    us: Array | None = None,
    key: jrandom.PRNGKeyArray | None = None,
) -> Array:
    """Batch version of SAPS Nonconformity Score for JAX arrays."""
    n_samples = probs.shape[0]

    if us is None:
        if key is None:
            us = jnp.array(np.random.default_rng().uniform(size=n_samples))
        else:
            key, subkey = jrandom.split(key)
            us = jrandom.uniform(subkey, shape=(n_samples,))

    max_probs = jnp.max(probs, axis=1)

    sorted_indices = jnp.argsort(-probs, axis=1)

    labels_expanded = labels[:, jnp.newaxis]

    rank_mask = sorted_indices == labels_expanded

    ranks = jnp.argmax(rank_mask.astype(jnp.int32), axis=1) + 1

    # Compute scores based on ranks
    scores = jnp.where(
        ranks == 1,
        us * max_probs,
        max_probs + lambda_val * (1 + us),
    )
    return scores


# Register the implementation
register(Array, saps_score_jax)
