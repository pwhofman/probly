"""JAX implementations of conformal quantile computation."""

from __future__ import annotations

from jax.core import Tracer
import jax.numpy as jnp

from ._common import calculate_quantile, calculate_weighted_quantile


@calculate_quantile.register((jnp.ndarray, Tracer))
def _jax_compute_quantile_score(scores: jnp.ndarray, alpha: float) -> float | jnp.ndarray:
    # Implementation for JAX arrays
    if not isinstance(alpha, Tracer) and not 0 <= alpha <= 1:
        msg = f"alpha must be in [0, 1], got {alpha}"
        raise ValueError(msg)

    n = len(scores)
    if n == 0:
        msg = "scores array is empty"
        raise ValueError(msg)

    q_level = jnp.ceil((n + 1) * (1 - alpha)) / n
    q_level = jnp.minimum(q_level, 1.0)  # ensure within [0, 1]

    # Inverted CDF / right-continuous step quantile
    # JAX does not support "inverted_cdf" method; "nearest" is the most precise available approximation.
    result = jnp.quantile(scores, q_level, method="nearest")
    # Preserve the eager Python-scalar API without concretizing traced results.
    return result if isinstance(result, Tracer) else float(result)


@calculate_weighted_quantile.register((jnp.ndarray, Tracer))
def _jax_compute_weighted_quantile(
    values: jnp.ndarray, quantile: float, sample_weight: jnp.ndarray | None = None
) -> float | jnp.ndarray:
    if sample_weight is None:
        result = jnp.quantile(values, quantile, method="linear")
        return result if isinstance(result, Tracer) else float(result)

    values = jnp.array(values)
    sample_weight = jnp.array(sample_weight)

    sorter = jnp.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = jnp.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= jnp.sum(sample_weight)

    result = jnp.interp(quantile, weighted_quantiles, values)
    return result if isinstance(result, Tracer) else float(result)
