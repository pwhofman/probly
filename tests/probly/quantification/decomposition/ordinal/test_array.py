"""Tests for ordinal decompositions on NumPy representations."""

from __future__ import annotations

import numpy as np
import pytest

from probly.quantification import (
    CategoricalVarianceDecomposition,
    GaussianVarianceDecomposition,
    LabelwiseBinaryEntropyDecomposition,
    LabelwiseBinaryVarianceDecomposition,
    OrdinalEntropyDecomposition,
    OrdinalVarianceDecomposition,
)
from probly.quantification.decomposition.ordinal import (
    categorical_variance_aleatoric,
    categorical_variance_total,
    gaussian_variance_aleatoric,
    gaussian_variance_epistemic,
    labelwise_binary_entropy_aleatoric,
    labelwise_binary_entropy_total,
    labelwise_binary_variance_aleatoric,
    labelwise_binary_variance_total,
    ordinal_binary_entropy_aleatoric,
    ordinal_binary_entropy_total,
    ordinal_binary_variance_aleatoric,
    ordinal_binary_variance_total,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)
from probly.representation.distribution.array_gaussian import (
    ArrayGaussianDistribution,
    ArrayGaussianDistributionSample,
)


def _categorical_sample() -> ArrayCategoricalDistributionSample:
    """Sample with shape (M=3, N=2, K=3) and sample_axis=0."""
    probs = np.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    return ArrayCategoricalDistributionSample(
        array=ArrayCategoricalDistribution(probs),
        sample_axis=0,
    )


def _constant_categorical_sample() -> ArrayCategoricalDistributionSample:
    """All three models agree: epistemic uncertainty should be zero."""
    probs = np.array(
        [
            [[0.70, 0.20, 0.10]],
            [[0.70, 0.20, 0.10]],
            [[0.70, 0.20, 0.10]],
        ],
        dtype=float,
    )
    return ArrayCategoricalDistributionSample(
        array=ArrayCategoricalDistribution(probs),
        sample_axis=0,
    )


def _gaussian_sample() -> ArrayGaussianDistributionSample:
    """Three Gaussian models with identical variance and different means."""
    gaussians = [
        ArrayGaussianDistribution(mean=np.array([m]), var=np.array([0.5]))
        for m in [1.0, 2.0, 3.0]
    ]
    return ArrayGaussianDistributionSample.from_iterable(gaussians, sample_axis=0)


def _identical_gaussian_sample() -> ArrayGaussianDistributionSample:
    """Three identical Gaussian models: epistemic uncertainty should be zero."""
    gaussians = [ArrayGaussianDistribution(mean=np.array([2.0]), var=np.array([0.5]))] * 3
    return ArrayGaussianDistributionSample.from_iterable(gaussians, sample_axis=0)


CATEGORICAL_DECOMP_CLASSES = [
    OrdinalEntropyDecomposition,
    OrdinalVarianceDecomposition,
    LabelwiseBinaryEntropyDecomposition,
    LabelwiseBinaryVarianceDecomposition,
    CategoricalVarianceDecomposition,
]


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_categorical_decomposition_shapes(cls) -> None:
    sample = _categorical_sample()
    d = cls(sample)
    assert d.total.shape == (2,)
    assert d.aleatoric.shape == (2,)
    assert d.epistemic.shape == (2,)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_categorical_decomposition_is_additive(cls) -> None:
    sample = _categorical_sample()
    d = cls(sample)
    np.testing.assert_allclose(d.total, d.aleatoric + d.epistemic, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_categorical_decomposition_epistemic_nonnegative(cls) -> None:
    sample = _categorical_sample()
    d = cls(sample)
    assert np.all(d.epistemic >= -1e-10)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_categorical_decomposition_total_ge_aleatoric(cls) -> None:
    sample = _categorical_sample()
    d = cls(sample)
    assert np.all(d.total >= d.aleatoric - 1e-10)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_categorical_decomposition_notion_access(cls) -> None:
    from probly.quantification.notion import AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty

    d = cls(_categorical_sample())
    assert isinstance(d["tu"], np.ndarray)
    assert isinstance(d["au"], np.ndarray)
    assert isinstance(d["eu"], np.ndarray)
    assert d[TotalUncertainty] is d.total
    assert d[AleatoricUncertainty] is d.aleatoric
    assert d[EpistemicUncertainty] is d.epistemic


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_constant_ensemble_has_zero_epistemic(cls) -> None:
    sample = _constant_categorical_sample()
    d = cls(sample)
    np.testing.assert_allclose(d.epistemic, 0.0, atol=1e-12)
    np.testing.assert_allclose(d.total, d.aleatoric, atol=1e-12)


def test_ordinal_entropy_vs_manual_ocs_formula() -> None:
    sample = _categorical_sample()
    p = sample.array.probabilities
    cum = np.cumsum(p, axis=-1)[..., :-1]
    p_bar = np.mean(cum, axis=0)

    def bh(x: np.ndarray) -> np.ndarray:
        from scipy.stats import entropy as scipy_entropy

        return scipy_entropy(np.stack([x, 1.0 - x], axis=-1), axis=-1)

    expected_tu = np.sum(bh(p_bar), axis=-1)
    expected_au = np.mean(np.sum(bh(cum), axis=-1), axis=0)

    d = OrdinalEntropyDecomposition(sample)
    np.testing.assert_allclose(d.total, expected_tu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d.aleatoric, expected_au, rtol=1e-12, atol=1e-12)


def test_ordinal_variance_vs_manual_ocs_formula() -> None:
    sample = _categorical_sample()
    p = sample.array.probabilities
    cum = np.cumsum(p, axis=-1)[..., :-1]
    p_bar = np.mean(cum, axis=0)

    expected_tu = np.sum(p_bar * (1.0 - p_bar), axis=-1)
    expected_au = np.mean(np.sum(cum * (1.0 - cum), axis=-1), axis=0)

    d = OrdinalVarianceDecomposition(sample)
    np.testing.assert_allclose(d.total, expected_tu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d.aleatoric, expected_au, rtol=1e-12, atol=1e-12)


def test_categorical_variance_vs_manual_formula() -> None:
    sample = _categorical_sample()
    p = sample.array.probabilities
    labels = np.arange(1, 4, dtype=float)

    p_bar = np.mean(p, axis=0)
    mu_bar = np.sum(labels * p_bar, axis=-1, keepdims=True)
    expected_tu = np.sum(((labels - mu_bar) ** 2) * p_bar, axis=-1)

    mu_m = np.sum(labels * p, axis=-1, keepdims=True)
    expected_au = np.mean(np.sum(((labels - mu_m) ** 2) * p, axis=-1), axis=0)

    d = CategoricalVarianceDecomposition(sample)
    np.testing.assert_allclose(d.total, expected_tu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d.aleatoric, expected_au, rtol=1e-12, atol=1e-12)


def test_standalone_measure_functions_match_decomposition() -> None:
    sample = _categorical_sample()

    d_ord_ent = OrdinalEntropyDecomposition(sample)
    np.testing.assert_allclose(ordinal_binary_entropy_total(sample), d_ord_ent.total)
    np.testing.assert_allclose(ordinal_binary_entropy_aleatoric(sample), d_ord_ent.aleatoric)

    d_ord_var = OrdinalVarianceDecomposition(sample)
    np.testing.assert_allclose(ordinal_binary_variance_total(sample), d_ord_var.total)
    np.testing.assert_allclose(ordinal_binary_variance_aleatoric(sample), d_ord_var.aleatoric)

    d_lw_ent = LabelwiseBinaryEntropyDecomposition(sample)
    np.testing.assert_allclose(labelwise_binary_entropy_total(sample), d_lw_ent.total)
    np.testing.assert_allclose(labelwise_binary_entropy_aleatoric(sample), d_lw_ent.aleatoric)

    d_lw_var = LabelwiseBinaryVarianceDecomposition(sample)
    np.testing.assert_allclose(labelwise_binary_variance_total(sample), d_lw_var.total)
    np.testing.assert_allclose(labelwise_binary_variance_aleatoric(sample), d_lw_var.aleatoric)

    d_cat_var = CategoricalVarianceDecomposition(sample)
    np.testing.assert_allclose(categorical_variance_total(sample), d_cat_var.total)
    np.testing.assert_allclose(categorical_variance_aleatoric(sample), d_cat_var.aleatoric)


def test_gaussian_variance_decomposition_values() -> None:
    sample = _gaussian_sample()
    d = GaussianVarianceDecomposition(sample)

    expected_au = np.array([0.5])
    expected_eu = np.var(np.array([1.0, 2.0, 3.0]), ddof=0)

    np.testing.assert_allclose(d.aleatoric, expected_au, rtol=1e-12)
    np.testing.assert_allclose(d.epistemic, expected_eu, rtol=1e-12)
    np.testing.assert_allclose(d.total, expected_au + expected_eu, rtol=1e-12)


def test_gaussian_variance_decomposition_is_additive() -> None:
    sample = _gaussian_sample()
    d = GaussianVarianceDecomposition(sample)
    np.testing.assert_allclose(d.total, d.aleatoric + d.epistemic, rtol=1e-12, atol=1e-12)


def test_identical_gaussian_has_zero_epistemic() -> None:
    sample = _identical_gaussian_sample()
    d = GaussianVarianceDecomposition(sample)
    np.testing.assert_allclose(d.epistemic, 0.0, atol=1e-12)


def test_gaussian_standalone_functions_match_decomposition() -> None:
    sample = _gaussian_sample()
    d = GaussianVarianceDecomposition(sample)
    np.testing.assert_allclose(gaussian_variance_aleatoric(sample), d.aleatoric)
    np.testing.assert_allclose(gaussian_variance_epistemic(sample), d.epistemic)


def test_ordinal_entropy_with_log_base() -> None:
    sample = _categorical_sample()
    d_nats = OrdinalEntropyDecomposition(sample, base=None)
    d_bits = OrdinalEntropyDecomposition(sample, base=2)
    d_norm = OrdinalEntropyDecomposition(sample, base="normalize")

    np.testing.assert_allclose(d_bits.total, d_nats.total / np.log(2), rtol=1e-12)
    np.testing.assert_allclose(d_norm.total, d_nats.total / np.log(2), rtol=1e-12)
