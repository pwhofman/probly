"""Utility functions for JAX models."""

from __future__ import annotations

import jax.numpy as jnp


def jax_entropy(p: jnp.ndarray) -> jnp.ndarray:
    """Shannon entropy H(p) computed in jax along the last dim; 0*log(0) treated as 0.

    The logarithm is fed with the zeros replaced by ones instead of masking its result, so the
    gradient stays finite for probability vectors that contain exact zeros.

    Args:
        p: Probabilities to compute entropy of.

    Returns:
        Entropy of probabilities p.
    """
    safe_p = jnp.where(p > 0, p, jnp.ones_like(p))
    result = -jnp.sum(p * jnp.log(safe_p), axis=-1)
    return jnp.clip(result, min=0.0) + 0.0


def intersection_probability(lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    """Intersection probability of a probability interval, per :cite:`wangCredalDeepEnsembles2024` Section 3.4.

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
