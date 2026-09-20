"""Jax implementations of sample-based uncertainty measures."""

from __future__ import annotations

import jax
from jax import numpy as jnp

from probly.quantification.measure.sample._common import (
    mean_squared_distance_to_scaled_one_hot,
    total_logit_sample_variance,
)
from probly.representation.distribution.jax_categorical import JaxCategoricalDistributionSample
from probly.representation.jax_functions import jax_mean, jax_sum


@mean_squared_distance_to_scaled_one_hot.register(JaxCategoricalDistributionSample)
def jax_mean_squared_distance_to_scaled_one_hot(
    sample: JaxCategoricalDistributionSample, scale: float | None = None
) -> jax.Array:
    r"""Numpy impl. uses :math:`\|h_k - s e_c\|^2 = \|h_k\|^2 - 2s \max_j h_{k,j} + s^2`(no one-hot built)."""
    array = sample.array.logits
    num_classes = array.shape[-1]
    target_scale = float(num_classes) if scale is None else float(scale)

    norm_sq = jax_sum(array * array, axis=-1)
    max_logit = jnp.max(array, axis=-1)
    per_member = norm_sq - 2.0 * target_scale * max_logit + target_scale * target_scale

    if sample.weights is not None:
        return jnp.average(per_member, axis=sample.sample_axis, weights=sample.weights)
    return jax_mean(per_member, axis=sample.sample_axis)


@total_logit_sample_variance.register(JaxCategoricalDistributionSample)
def jax_total_logit_sample_variance(sample: JaxCategoricalDistributionSample) -> jax.Array:
    """Jax impl. Variance of total logits (logits summed across members)."""
    array = sample.array.logits
    sample_axis = sample.sample_axis
    return jnp.var(array, axis=sample_axis).sum(axis=-1)
