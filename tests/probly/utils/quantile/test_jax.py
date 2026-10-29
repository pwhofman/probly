"""Tests for the jax backend of ``calculate_quantile``."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from probly.utils.quantile import calculate_quantile, calculate_weighted_quantile


class TestQuantileJax:
    """`calculate_quantile` for jax arrays."""

    def test_jax_quantile_basic(self) -> None:
        scores = jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
        q = calculate_quantile(scores, alpha=0.1)
        assert isinstance(q, float)

    def test_jax_alpha_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be in"):
            calculate_quantile(jnp.asarray([0.1, 0.2]), alpha=1.5)

    def test_jax_empty_scores_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            calculate_quantile(jnp.asarray([]), alpha=0.1)

    def test_jax_weighted_quantile_unweighted(self) -> None:
        values = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        assert calculate_weighted_quantile(values, 0.5) == pytest.approx(3.0)

    def test_jax_weighted_quantile_with_weights(self) -> None:
        values = jnp.asarray([1.0, 2.0, 3.0])
        weights = jnp.asarray([1.0, 0.0, 0.0])
        result = calculate_weighted_quantile(values, 0.5, sample_weight=weights)
        assert result == pytest.approx(1.0)


@pytest.mark.parametrize("weighted", [False, True])
def test_quantiles_support_transformed_calls(weighted):
    scores = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    if weighted:

        def quantile(values, level):
            return calculate_weighted_quantile(values, level, jnp.ones_like(values))
    else:
        quantile = calculate_quantile

    actual = jax.jit(quantile)(scores, 0.5)
    assert jnp.allclose(actual, quantile(scores, 0.5))
    mapped = jax.jit(jax.vmap(lambda values: quantile(values, 0.5)))(jnp.stack([scores, scores + 1]))
    assert jnp.allclose(mapped, jnp.array([actual, actual + 1]))
    gradient = jax.jit(jax.grad(lambda values: quantile(values, 0.5)))(scores)
    assert jnp.isfinite(gradient).all()
