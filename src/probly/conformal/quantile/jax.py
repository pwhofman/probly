"""This module contains the quantile-based conformal prediction methods."""

from ._common import calculate_quantile, calculate_weighted_quantile

import jax.numpy as jnp


@calculate_quantile.register(jnp.ndarray)
def _compute_quantile_score_jax(scores: jnp.ndarray, alpha: float) -> jnp.ndarray:
    # Implementation for JAX arrays
    if not 0 <= alpha <= 1:
        msg = f"alpha must be in [0, 1], got {alpha}"
        raise ValueError(msg)

    n = len(scores)
    if n == 0:
        msg = "scores array is empty"
        raise ValueError(msg)

    q_level = jnp.ceil((n + 1) * (1 - alpha)) / n
    q_level = jnp.minimum(q_level, 1.0)  # ensure within [0, 1]

    # Inverted CDF / right-continuous step quantile
    return float(
        jnp.quantile(scores, q_level, method="nearest")
    )  # JAX does not support "inverted_cdf" method, using "nearest" as an approximation, which is the most precise method available in JAX for quantiles.


@calculate_weighted_quantile.register(jnp.ndarray)
def _compute_weighted_quantile_jax(
    values: jnp.ndarray, quantile: float, sample_weight: jnp.ndarray | None = None
) -> jnp.ndarray:
    if sample_weight is None:
        return float(jnp.quantile(values, quantile, method="linear"))

    values = jnp.array(values)
    sample_weight = jnp.array(sample_weight)

    sorter = jnp.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = jnp.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= jnp.sum(sample_weight)

    return float(jnp.interp(quantile, weighted_quantiles, values))