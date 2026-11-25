"""Utility functions for JAX models."""

from __future__ import annotations

import jax
from jax.core import Tracer
import jax.numpy as jnp
import numpy as np

from ._common import elementwise_entropy, entropy, intersection_probability


def jax_fresh_prng_key() -> jax.Array:
    """Draw a fresh PRNG key seeded from OS entropy.

    Mirrors the numpy backend, where a sampler without an explicit generator falls back to a
    fresh ``numpy.random.default_rng()``, so repeated calls produce different draws.

    Returns:
        A new PRNG key.
    """
    seed = int(np.random.default_rng().integers(np.iinfo(np.uint32).max, dtype=np.uint32))
    return jax.random.key(seed)


@elementwise_entropy.register((jax.Array, Tracer))
def jax_elementwise_entropy(p: jnp.ndarray) -> jnp.ndarray:
    """Entropy contributions ``-p * log(p)`` computed in jax; 0*log(0) treated as 0.

    The logarithm is fed with the zeros replaced by ones instead of masking its result, so the
    gradient stays finite for probabilities that are exactly zero.

    Args:
        p: Probabilities to compute the entropy contributions of.

    Returns:
        Entropy contributions of the shape of ``p``.
    """
    safe_p = jnp.where(p > 0, p, jnp.ones_like(p))
    return -(p * jnp.log(safe_p))


@entropy.register((jax.Array, Tracer))
def jax_entropy(p: jnp.ndarray) -> jnp.ndarray:
    """Shannon entropy H(p) computed in jax along the last dim; 0*log(0) treated as 0.

    Args:
        p: Probabilities to compute entropy of.

    Returns:
        Entropy of probabilities p.
    """
    result = jnp.sum(jax_elementwise_entropy(p), axis=-1)
    return jnp.clip(result, min=0.0) + 0.0


@intersection_probability.register((jax.Array, Tracer))
def jax_intersection_probability(lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    """Intersection probability of a probability interval, per :cite:`wangCreINNsCredalSet2025` Section 3.4.

    Reduces an interval credal set ``[lower, upper]`` to a single probability
    vector by ``q_int_k = lower_k + alpha * (upper_k - lower_k)`` with
    ``alpha = (1 - sum(lower)) / sum(upper - lower)``. The implementation
    handles the degenerate case ``upper == lower`` (zero width) by returning
    ``lower`` directly, avoiding ``0 / 0`` and keeping autodiff well-defined.

    Args:
        lower: Lower bounds of shape ``(..., num_classes)``.
        upper: Upper bounds of shape ``(..., num_classes)``.

    Returns:
        Intersection probability array of shape ``(..., num_classes)``.
    """
    slack = upper - lower
    slack_sum = jnp.sum(slack, axis=-1, keepdims=True)
    remaining = 1 - jnp.sum(lower, axis=-1, keepdims=True)
    remaining = jnp.clip(remaining, min=0)
    denominator = jnp.where(slack_sum != 0, slack_sum, jnp.ones_like(slack_sum))
    weights = jnp.where(slack_sum != 0, slack / denominator, jnp.zeros_like(slack))
    return lower + remaining * weights
