"""JAX implementation for Inner Product scores."""

from __future__ import annotations

import jax
from jax.core import Tracer
import jax.numpy as jnp

from probly.representation.distribution import CategoricalDistribution
from probly.representation.distribution.jax_categorical import JaxCategoricalDistribution
from probly.representation.sample.jax import JaxArraySample

from ._common import inner_product_score_func


@inner_product_score_func.register((jax.Array, Tracer))
def compute_inner_product_score_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Computes the Inner Product score using JAX Array."""
    y_pred_t = jnp.asarray(y_pred)
    distribution_target = isinstance(y_true, CategoricalDistribution)
    y_true_t = jnp.asarray(y_true.probabilities if distribution_target else y_true)
    if y_pred_t.ndim == 0:
        msg = "Predicted probabilities must have a class axis."
        raise ValueError(msg)

    if not distribution_target and jnp.issubdtype(y_true_t.dtype, jnp.integer):
        batch_shape = jnp.broadcast_shapes(y_pred_t.shape[:-1], y_true_t.shape)
        probabilities = jnp.broadcast_to(y_pred_t, (*batch_shape, y_pred_t.shape[-1]))
        labels = jnp.broadcast_to(y_true_t, batch_shape)
        return 1.0 - jnp.take_along_axis(probabilities, labels[..., None], axis=-1).squeeze(-1)

    if not distribution_target and not jnp.issubdtype(y_true_t.dtype, jnp.floating):
        msg = "Targets must be integer labels, floating-point probabilities, or a categorical distribution."
        raise TypeError(msg)
    if y_true_t.ndim == 0 or y_true_t.shape[-1] != y_pred_t.shape[-1]:
        msg = "Target probabilities must have the same number of classes as predictions."
        raise ValueError(msg)

    return 1.0 - jnp.sum(y_pred_t * y_true_t, axis=-1)


@inner_product_score_func.register(JaxArraySample)
def _(y_pred: JaxArraySample, y_true: jax.Array) -> jax.Array:
    """Compute memberwise scores for JAX samples."""
    return inner_product_score_func(y_pred.array, y_true)


@inner_product_score_func.register(JaxCategoricalDistribution)
def _(y_pred: JaxCategoricalDistribution, y_true: jax.Array) -> jax.Array:
    """Compute the score from normalized categorical probabilities."""
    return inner_product_score_func(y_pred.probabilities, y_true)
