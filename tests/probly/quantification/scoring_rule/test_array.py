"""Tests for proper scoring rule loss vectors on NumPy arrays."""

from __future__ import annotations

import numpy as np

from probly.quantification.scoring_rule import BrierLoss, LogLoss, SphericalLoss, ZeroOneLoss


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
