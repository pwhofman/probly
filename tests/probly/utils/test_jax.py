"""Tests for utils.jax functions."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from probly.utils import elementwise_entropy, entropy, intersection_probability, machine_epsilon
from probly.utils.jax import jax_elementwise_entropy, jax_entropy, jax_intersection_probability, jax_machine_epsilon


def test_entropy_dispatches_to_jax() -> None:
    p = jnp.array([[0.5, 0.5], [1.0, 0.0]])
    assert jnp.allclose(entropy(p), jax_entropy(p))
    assert jnp.allclose(entropy(p), jnp.array([math.log(2.0), 0.0]))


def test_elementwise_entropy_dispatches_to_jax() -> None:
    p = jnp.array([0.5, 1.0, 0.0, -1e-9])
    assert jnp.allclose(elementwise_entropy(p), jax_elementwise_entropy(p))
    # Zero and negative entries contribute nothing.
    assert jnp.allclose(elementwise_entropy(p), jnp.array([0.5 * math.log(2.0), 0.0, 0.0, 0.0]))


def test_intersection_probability_dispatches_to_jax() -> None:
    lower = jnp.array([[0.2, 0.3]])
    upper = jnp.array([[0.6, 0.7]])
    result = intersection_probability(lower, upper)
    assert jnp.allclose(result, jax_intersection_probability(lower, upper))
    assert jnp.allclose(result.sum(-1), 1.0)


def test_machine_epsilon_dispatches_to_jax() -> None:
    values = jnp.ones(2, dtype=jnp.float32)
    assert machine_epsilon(values) == jax_machine_epsilon(values) == float(jnp.finfo(jnp.float32).eps)
    assert isinstance(machine_epsilon(values), float)


def test_dispatch_inside_jit() -> None:
    p = jnp.array([[0.25, 0.75]])
    assert jnp.allclose(jax.jit(entropy)(p), jax_entropy(p))
    # The dtype of a traced array is known while tracing, so its machine epsilon is a constant of the compiled function.
    assert jnp.allclose(jax.jit(lambda x: x * machine_epsilon(x))(p), p * float(jnp.finfo(p.dtype).eps))
