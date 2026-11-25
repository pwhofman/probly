"""Jax implementations of regression variance measures."""

from __future__ import annotations

from jax import numpy as jnp

from probly.representation.sample.jax import JaxSample

from ._common import LogBase, expected_conditional_variance


@expected_conditional_variance.register(JaxSample)
def jax_sample_expected_conditional_variance(
    sample: JaxSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the expected conditional variance of a raw jax array sample (zero for point predictions)."""
    return jnp.zeros_like(jnp.mean(sample.array, axis=sample.sample_axis))
