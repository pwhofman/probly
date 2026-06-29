"""Tests for the scoring-rule decomposition on NumPy representations."""

from __future__ import annotations

import numpy as np

from probly.quantification import (
    BrierLoss,
    LogLoss,
    SecondOrderEntropyDecomposition,
    SecondOrderScoringRuleDecomposition,
    SecondOrderZeroOneDecomposition,
    SphericalLoss,
    ZeroOneLoss,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistributionSample,
    ArrayProbabilityCategoricalDistribution,
)

_PROBS = np.array(
    [
        [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
        [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
        [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
    ],
    dtype=float,
)


def _sample() -> ArrayCategoricalDistributionSample:
    return ArrayCategoricalDistributionSample(
        array=ArrayProbabilityCategoricalDistribution(_PROBS),
        sample_axis=0,
    )


def test_log_loss_matches_entropy_decomposition() -> None:
    sr = SecondOrderScoringRuleDecomposition(_sample(), LogLoss())
    ent = SecondOrderEntropyDecomposition(_sample())
    np.testing.assert_allclose(sr.total, ent.total, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sr.aleatoric, ent.aleatoric, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sr.epistemic, ent.epistemic, rtol=1e-10, atol=1e-12)


def test_zero_one_loss_matches_zero_one_decomposition() -> None:
    sr = SecondOrderScoringRuleDecomposition(_sample(), ZeroOneLoss())
    zo = SecondOrderZeroOneDecomposition(_sample())
    np.testing.assert_allclose(sr.total, zo.total, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sr.aleatoric, zo.aleatoric, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sr.epistemic, zo.epistemic, rtol=1e-12, atol=1e-12)


def test_brier_loss_matches_gini_closed_form() -> None:
    mean = _PROBS.mean(axis=0)
    tu = 1.0 - np.sum(mean**2, axis=-1)
    au = np.mean(1.0 - np.sum(_PROBS**2, axis=-1), axis=0)
    sr = SecondOrderScoringRuleDecomposition(_sample(), BrierLoss())
    np.testing.assert_allclose(sr.total, tu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sr.aleatoric, au, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sr.epistemic, tu - au, rtol=1e-12, atol=1e-12)


def test_spherical_loss_matches_closed_form_and_is_additive() -> None:
    mean = _PROBS.mean(axis=0)
    tu = 1.0 - np.sqrt(np.sum(mean**2, axis=-1))
    au = np.mean(1.0 - np.sqrt(np.sum(_PROBS**2, axis=-1)), axis=0)
    sr = SecondOrderScoringRuleDecomposition(_sample(), SphericalLoss())
    np.testing.assert_allclose(sr.total, tu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sr.aleatoric, au, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sr.total, sr.aleatoric + sr.epistemic, rtol=1e-12, atol=1e-12)
    assert np.all(sr.epistemic >= -1e-12)
