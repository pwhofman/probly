"""Tests for proper scoring rule loss vectors on NumPy arrays."""

from __future__ import annotations

import numpy as np
import pytest

from probly.quantification.scoring_rule import BrierLoss, LogLoss, SphericalLoss, ZeroOneLoss
from probly.representation.distribution.array_categorical import ArrayLogitCategoricalDistribution
from probly.representation.sample.array import ArraySample


@pytest.mark.parametrize("rule", [LogLoss(), BrierLoss(), ZeroOneLoss(), SphericalLoss()])
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_representation_loss(rule, sample_axis: int) -> None:
    probabilities = np.broadcast_to([0.2, 0.3, 0.1, 0.4], (2, 3, 4)).copy()
    distribution = ArrayLogitCategoricalDistribution(np.log(probabilities) + 3.0)
    expected = rule.loss(probabilities)
    result = rule.loss(distribution)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    weights = np.arange(1, probabilities.shape[sample_axis] + 1, dtype=float)
    for values in (probabilities, distribution):
        sample = ArraySample(values, sample_axis=sample_axis, weights=weights)
        result = rule.loss(sample)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == sample_axis
        assert result.weights is weights
        np.testing.assert_allclose(result.array, expected, atol=1e-6)


def test_log_loss_vector() -> None:
    p = np.array([[0.5, 0.5], [0.25, 0.75]])
    np.testing.assert_allclose(LogLoss().loss(p), -np.log(p), rtol=1e-12, atol=1e-12)


def test_brier_loss_vector() -> None:
    # At a vertex the Brier loss is 0 for the true label and 2 for the other.
    p = np.array([[1.0, 0.0]])
    np.testing.assert_allclose(BrierLoss().loss(p), np.array([[0.0, 2.0]]), rtol=1e-12, atol=1e-12)


def test_zero_one_loss_vector() -> None:
    p = np.array([[0.7, 0.3], [0.2, 0.8]])
    np.testing.assert_allclose(ZeroOneLoss().loss(p), np.array([[0.0, 1.0], [1.0, 0.0]]), rtol=1e-12, atol=1e-12)


def test_spherical_loss_vector() -> None:
    p = np.array([[1.0, 0.0]])
    np.testing.assert_allclose(SphericalLoss().loss(p), np.array([[0.0, 1.0]]), rtol=1e-12, atol=1e-12)


def test_loss_preserves_shape() -> None:
    p = np.full((4, 3, 5), 1.0 / 5.0)
    for rule in (LogLoss(), BrierLoss(), ZeroOneLoss(), SphericalLoss()):
        assert rule.loss(p).shape == p.shape


def test_scoring_rules_are_value_equal() -> None:
    assert BrierLoss() == BrierLoss()
    assert LogLoss() != BrierLoss()
