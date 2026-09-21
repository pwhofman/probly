"""Tests for utils.jax functions."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from probly.utils import entropy, intersection_probability
from probly.utils.jax import jax_entropy, jax_intersection_probability


def test_entropy_dispatches_to_jax() -> None:
    p = jnp.array([[0.5, 0.5], [1.0, 0.0]])
    assert jnp.allclose(entropy(p), jax_entropy(p))
    assert jnp.allclose(entropy(p), jnp.array([math.log(2.0), 0.0]))


def test_intersection_probability_dispatches_to_jax() -> None:
    lower = jnp.array([[0.2, 0.3]])
    upper = jnp.array([[0.6, 0.7]])
    result = intersection_probability(lower, upper)
    assert jnp.allclose(result, jax_intersection_probability(lower, upper))
    assert jnp.allclose(result.sum(-1), 1.0)


def test_dispatch_inside_jit() -> None:
    p = jnp.array([[0.25, 0.75]])
    assert jnp.allclose(jax.jit(entropy)(p), jax_entropy(p))
