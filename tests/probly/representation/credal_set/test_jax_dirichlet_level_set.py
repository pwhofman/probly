"""Tests for jax-backed Dirichlet level set credal sets."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from probly.representation.credal_set._common import create_dirichlet_level_set_credal_set
from probly.representation.credal_set.jax import JaxDirichletLevelSetCredalSet


def _make_credal_set(alphas: list[float], threshold: float) -> JaxDirichletLevelSetCredalSet:
    return JaxDirichletLevelSetCredalSet(
        alphas=jnp.array(alphas, dtype=float),
        threshold=jnp.array(threshold, dtype=float),
    )


def test_construction() -> None:
    cs = _make_credal_set([5.0, 3.0, 2.0], 0.5)
    assert isinstance(cs, JaxDirichletLevelSetCredalSet)
    assert cs.num_classes == 3


def test_barycenter_is_dirichlet_mean() -> None:
    cs = _make_credal_set([6.0, 3.0, 1.0], 0.5)
    expected = jnp.array([0.6, 0.3, 0.1], dtype=float)
    assert jnp.allclose(cs.barycenter.probabilities, expected, atol=1e-6)


def test_lower_upper_valid() -> None:
    """Lower bounds should be <= upper bounds and within [0, 1]."""
    cs = _make_credal_set([5.0, 3.0, 2.0], 0.5)
    lower = cs.lower()
    upper = cs.upper()
    assert jnp.all(lower >= 0.0)
    assert jnp.all(upper <= 1.0)
    assert jnp.all(lower <= upper + 1e-6)


def test_high_threshold_tight_bounds() -> None:
    """With threshold near 1, bounds should be tight around the mode."""
    cs = _make_credal_set([10.0, 5.0, 3.0], 0.99)
    lower = cs.lower()
    upper = cs.upper()
    width = upper - lower
    assert jnp.all(width < 0.3)


def test_low_threshold_wide_bounds() -> None:
    """With threshold near 0, bounds should cover most of the simplex."""
    cs = _make_credal_set([5.0, 3.0, 2.0], 0.01)
    lower = cs.lower()
    upper = cs.upper()
    width = upper - lower
    assert jnp.any(width > 0.3)


def test_batch_shape_preserved() -> None:
    """Batch dimensions should be preserved."""
    alphas = jnp.array([[5.0, 3.0, 2.0], [10.0, 1.0, 1.0]], dtype=float)
    threshold = jnp.array([0.5, 0.5], dtype=float)
    cs = JaxDirichletLevelSetCredalSet(alphas=alphas, threshold=threshold)
    assert cs.lower().shape == (2, 3)
    assert cs.upper().shape == (2, 3)
    assert cs.barycenter.shape == (2,)


def test_factory_creates_correct_type() -> None:
    """Factory function should create JaxDirichletLevelSetCredalSet."""
    alphas = jnp.array([5.0, 3.0, 2.0], dtype=float)
    result = create_dirichlet_level_set_credal_set(alphas, 0.5)
    assert isinstance(result, JaxDirichletLevelSetCredalSet)


def test_explicit_key_gives_reproducible_bounds() -> None:
    """Passing an explicit key should make lower()/upper() deterministic and repeatable."""
    cs = _make_credal_set([5.0, 3.0, 2.0], 0.5)
    key = jax.random.key(42)

    lower_a = cs.lower(key=key)
    lower_b = cs.lower(key=key)
    upper_a = cs.upper(key=key)
    upper_b = cs.upper(key=key)

    assert jnp.array_equal(lower_a, lower_b)
    assert jnp.array_equal(upper_a, upper_b)

    other_key = jax.random.key(7)
    lower_other = cs.lower(key=other_key)
    assert lower_other.shape == lower_a.shape
