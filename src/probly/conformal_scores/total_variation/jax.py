"""JAX implementation for Total Variation scores."""

from __future__ import annotations

import jax
from jax.core import Tracer
import jax.numpy as jnp

from probly.representation.distribution import CategoricalDistribution
from probly.representation.distribution.jax_categorical import JaxCategoricalDistribution
from probly.representation.sample.jax import JaxArraySample

from ._common import tv_score_func


@tv_score_func.register((jax.Array, Tracer))
def compute_tv_score_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Computes the Total Variation score using JAX arrays."""
    y_pred_j = jnp.asarray(y_pred)
    distribution_target = isinstance(y_true, CategoricalDistribution)
    y_true_j = jnp.asarray(y_true.probabilities if distribution_target else y_true)
    if y_pred_j.ndim == 0:
        msg = "Predicted probabilities must have a class axis."
        raise ValueError(msg)

    if not distribution_target and jnp.issubdtype(y_true_j.dtype, jnp.integer):
        batch_shape = jnp.broadcast_shapes(y_pred_j.shape[:-1], y_true_j.shape)
        probabilities = jnp.broadcast_to(y_pred_j, (*batch_shape, y_pred_j.shape[-1]))
        labels = jnp.broadcast_to(y_true_j, batch_shape)
        selected = jnp.take_along_axis(probabilities, labels[..., None], axis=-1).squeeze(-1)
        # Replace the selected class's contribution without allocating a one-hot target.
        return 0.5 * (jnp.abs(y_pred_j).sum(axis=-1) - jnp.abs(selected) + jnp.abs(selected - 1.0))

    if not distribution_target and not jnp.issubdtype(y_true_j.dtype, jnp.floating):
        msg = "Targets must be integer labels, floating-point probabilities, or a categorical distribution."
        raise TypeError(msg)
    if y_true_j.ndim == 0 or y_true_j.shape[-1] != y_pred_j.shape[-1]:
        msg = "Target probabilities must have the same number of classes as predictions."
        raise ValueError(msg)

    return 0.5 * jnp.sum(jnp.abs(y_pred_j - y_true_j), axis=-1)


@tv_score_func.register(JaxArraySample)
def _(y_pred: JaxArraySample, y_true: jax.Array) -> jax.Array:
    """Compute total variation scores for JAX samples."""
    return tv_score_func(y_pred.array, y_true)


@tv_score_func.register(JaxCategoricalDistribution)
def _(y_pred: JaxCategoricalDistribution, y_true: jax.Array) -> jax.Array:
    """Compute total variation scores for JAX categorical distributions."""
    return tv_score_func(y_pred.probabilities, y_true)
