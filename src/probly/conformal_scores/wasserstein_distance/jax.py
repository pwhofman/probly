"""JAX implementation for Wasserstein distance scores."""

from __future__ import annotations

import jax
from jax.core import Tracer
import jax.numpy as jnp

from probly.representation.distribution import CategoricalDistribution
from probly.representation.distribution.jax_categorical import JaxCategoricalDistribution
from probly.representation.sample.jax import JaxArraySample

from ._common import wasserstein_distance_score_func


@wasserstein_distance_score_func.register((jax.Array, Tracer))
def compute_wasserstein_distance_score_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Computes the Wasserstein distance score using JAX arrays.

    Args:
        y_pred: Predicted probability mass functions.
        y_true: True probability mass functions or integer labels.
    """
    y_pred_j = jnp.asarray(y_pred)
    distribution_target = isinstance(y_true, CategoricalDistribution)
    y_true_j = jnp.asarray(y_true.probabilities if distribution_target else y_true)
    if y_pred_j.ndim == 0:
        msg = "Predicted probabilities must have a class axis."
        raise ValueError(msg)

    if not distribution_target and jnp.issubdtype(y_true_j.dtype, jnp.integer):
        # A point mass has a step-function CDF; no one-hot array or target cumsum is needed.
        target_cdf = jnp.arange(y_pred_j.shape[-1]) >= y_true_j[..., None]
        return jnp.sum(jnp.abs(jnp.cumsum(y_pred_j, axis=-1) - target_cdf), axis=-1)

    if not distribution_target and not jnp.issubdtype(y_true_j.dtype, jnp.floating):
        msg = "Targets must be integer labels, floating-point probabilities, or a categorical distribution."
        raise TypeError(msg)
    if y_true_j.ndim == 0 or y_true_j.shape[-1] != y_pred_j.shape[-1]:
        msg = "Target probabilities must have the same number of classes as predictions."
        raise ValueError(msg)

    return jnp.sum(jnp.abs(jnp.cumsum(y_pred_j, axis=-1) - jnp.cumsum(y_true_j, axis=-1)), axis=-1)


@wasserstein_distance_score_func.register(JaxArraySample)
def _(y_pred: JaxArraySample, y_true: jax.Array) -> jax.Array:
    """Compute Wasserstein distance scores for JAX samples."""
    return wasserstein_distance_score_func(y_pred.array, y_true)


@wasserstein_distance_score_func.register(JaxCategoricalDistribution)
def _(y_pred: JaxCategoricalDistribution, y_true: jax.Array) -> jax.Array:
    """Compute Wasserstein distance scores for JAX categorical distributions."""
    return wasserstein_distance_score_func(y_pred.probabilities, y_true)
