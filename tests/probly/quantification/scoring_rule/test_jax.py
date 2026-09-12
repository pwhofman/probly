"""Tests for proper scoring rule loss vectors on JAX arrays."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp

from probly.quantification.scoring_rule import BrierLoss, LogLoss, SphericalLoss, ZeroOneLoss
from probly.representation.distribution.jax_categorical import JaxLogitCategoricalDistribution
from probly.representation.sample.jax import JaxArraySample


@pytest.mark.parametrize("rule", [LogLoss(), BrierLoss(), ZeroOneLoss(), SphericalLoss()])
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_representation_loss(rule, sample_axis: int) -> None:
    probabilities = jnp.broadcast_to(jnp.array([0.2, 0.3, 0.1, 0.4]), (2, 3, 4))
    distribution = JaxLogitCategoricalDistribution(jnp.log(probabilities) + 3.0)
    expected = rule.loss(probabilities)
    result = rule.loss(distribution)
    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, expected, atol=1e-6)
    weights = jnp.arange(1, probabilities.shape[sample_axis] + 1, dtype=float)
    for values in (probabilities, distribution):
        sample = JaxArraySample(values, sample_axis=sample_axis, weights=weights)
        result = rule.loss(sample)
        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == sample_axis
        assert result.weights is weights
        assert jnp.allclose(result.array, expected, atol=1e-6)


def test_log_loss_vector() -> None:
    p = jnp.array([[0.5, 0.5], [0.25, 0.75]])
    assert jnp.allclose(LogLoss().loss(p), -jnp.log(p), rtol=1e-6, atol=1e-6)


def test_brier_loss_vector() -> None:
    # At a vertex the Brier loss is 0 for the true label and 2 for the other.
    p = jnp.array([[1.0, 0.0]])
    assert jnp.allclose(BrierLoss().loss(p), jnp.array([[0.0, 2.0]]), rtol=1e-6, atol=1e-6)


def test_zero_one_loss_vector() -> None:
    p = jnp.array([[0.7, 0.3], [0.2, 0.8]])
    assert jnp.allclose(ZeroOneLoss().loss(p), jnp.array([[0.0, 1.0], [1.0, 0.0]]), rtol=1e-6, atol=1e-6)


def test_spherical_loss_vector() -> None:
    p = jnp.array([[1.0, 0.0]])
    assert jnp.allclose(SphericalLoss().loss(p), jnp.array([[0.0, 1.0]]), rtol=1e-6, atol=1e-6)


def test_loss_preserves_shape() -> None:
    p = jnp.full((4, 3, 5), 1.0 / 5.0)
    for rule in (LogLoss(), BrierLoss(), ZeroOneLoss(), SphericalLoss()):
        assert rule.loss(p).shape == p.shape


def test_scoring_rules_are_value_equal() -> None:
    assert BrierLoss() == BrierLoss()
    assert LogLoss() != BrierLoss()
