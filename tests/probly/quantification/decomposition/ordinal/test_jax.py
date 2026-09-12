"""Tests for ordinal decompositions on Jax representations."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
from jax import numpy as jnp
from scipy.stats import entropy as scipy_entropy

from probly.quantification import (
    CategoricalVarianceDecomposition,
    LabelwiseBinaryEntropyDecomposition,
    LabelwiseBinaryVarianceDecomposition,
    OrdinalEntropyDecomposition,
    OrdinalVarianceDecomposition,
    SecondOrderVarianceDecomposition,
)
from probly.quantification.decomposition.ordinal import (
    categorical_variance_aleatoric,
    categorical_variance_total,
    conditional_variance,
    labelwise_conditional_entropy,
    labelwise_conditional_variance,
    labelwise_entropy_of_expected_predictive_distribution,
    labelwise_variance_of_expected_predictive_distribution,
    mutual_information_variance,
    ordinal_conditional_entropy,
    ordinal_conditional_variance,
    ordinal_entropy_of_expected_predictive_distribution,
    ordinal_variance_of_expected_predictive_distribution,
)
from probly.quantification.measure.ordinal import labelwise_entropy, labelwise_variance
from probly.quantification.notion import AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistributionSample,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.distribution.jax_gaussian import (
    JaxGaussianDistribution,
    JaxGaussianDistributionSample,
)


def _categorical_sample() -> JaxCategoricalDistributionSample:
    """Sample with shape (M=3, N=2, K=3) and sample_axis=0."""
    probs = jnp.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    return JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probs),
        sample_axis=0,
    )


def _constant_categorical_sample() -> JaxCategoricalDistributionSample:
    """All three models agree: epistemic uncertainty should be zero."""
    probs = jnp.array(
        [
            [[0.70, 0.20, 0.10]],
            [[0.70, 0.20, 0.10]],
            [[0.70, 0.20, 0.10]],
        ],
        dtype=float,
    )
    return JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probs),
        sample_axis=0,
    )


def _gaussian_sample() -> JaxGaussianDistributionSample:
    """Three Gaussian models with identical variance and different means."""
    gaussians = [JaxGaussianDistribution(mean=jnp.array([m]), var=jnp.array([0.5])) for m in [1.0, 2.0, 3.0]]
    return JaxGaussianDistributionSample.from_iterable(gaussians, sample_axis=0)


def _identical_gaussian_sample() -> JaxGaussianDistributionSample:
    """Three identical Gaussian models: epistemic uncertainty should be zero."""
    gaussians = [JaxGaussianDistribution(mean=jnp.array([2.0]), var=jnp.array([0.5]))] * 3
    return JaxGaussianDistributionSample.from_iterable(gaussians, sample_axis=0)


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
    assert jnp.allclose(d.total, d.aleatoric + d.epistemic, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_categorical_decomposition_epistemic_nonnegative(cls) -> None:
    sample = _categorical_sample()
    d = cls(sample)
    assert jnp.allclose(d.epistemic, jnp.maximum(d.epistemic, -1e-10), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_categorical_decomposition_total_ge_aleatoric(cls) -> None:
    sample = _categorical_sample()
    d = cls(sample)
    assert jnp.allclose(d.total, jnp.maximum(d.total, d.aleatoric - 1e-10), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_categorical_decomposition_notion_access(cls) -> None:
    d = cls(_categorical_sample())
    assert isinstance(d["tu"], jnp.ndarray)
    assert isinstance(d["au"], jnp.ndarray)
    assert isinstance(d["eu"], jnp.ndarray)
    assert d[TotalUncertainty] is d.total
    assert d[AleatoricUncertainty] is d.aleatoric
    assert d[EpistemicUncertainty] is d.epistemic


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_constant_ensemble_has_zero_epistemic(cls) -> None:
    sample = _constant_categorical_sample()
    d = cls(sample)
    assert jnp.allclose(d.epistemic, 0.0, atol=1e-6)
    assert jnp.allclose(d.total, d.aleatoric, atol=1e-6)


def test_ordinal_entropy_vs_manual_ocs_formula() -> None:
    sample = _categorical_sample()
    p = sample.array.probabilities
    cum = jnp.cumsum(p, axis=-1)[..., :-1]
    p_bar = jnp.mean(cum, axis=0)

    def bh(x: jnp.ndarray) -> jnp.ndarray:
        return scipy_entropy(jnp.stack([x, 1.0 - x], axis=-1), axis=-1)

    expected_tu = jnp.sum(bh(p_bar), axis=-1)
    expected_au = jnp.mean(jnp.sum(bh(cum), axis=-1), axis=0)

    d = OrdinalEntropyDecomposition(sample)
    assert jnp.allclose(d.total, expected_tu, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(d.aleatoric, expected_au, rtol=1e-6, atol=1e-6)


def test_ordinal_variance_vs_manual_ocs_formula() -> None:
    sample = _categorical_sample()
    p = sample.array.probabilities
    cum = jnp.cumsum(p, axis=-1)[..., :-1]
    p_bar = jnp.mean(cum, axis=0)

    expected_tu = jnp.sum(p_bar * (1.0 - p_bar), axis=-1)
    expected_au = jnp.mean(jnp.sum(cum * (1.0 - cum), axis=-1), axis=0)

    d = OrdinalVarianceDecomposition(sample)
    assert jnp.allclose(d.total, expected_tu, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(d.aleatoric, expected_au, rtol=1e-6, atol=1e-6)


def test_categorical_variance_vs_manual_formula() -> None:
    sample = _categorical_sample()
    p = sample.array.probabilities
    labels = jnp.arange(1, 4, dtype=float)

    p_bar = jnp.mean(p, axis=0)
    mu_bar = jnp.sum(labels * p_bar, axis=-1, keepdims=True)
    expected_tu = jnp.sum(((labels - mu_bar) ** 2) * p_bar, axis=-1)

    mu_m = jnp.sum(labels * p, axis=-1, keepdims=True)
    expected_au = jnp.mean(jnp.sum(((labels - mu_m) ** 2) * p, axis=-1), axis=0)

    d = CategoricalVarianceDecomposition(sample)
    assert jnp.allclose(d.total, expected_tu, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(d.aleatoric, expected_au, rtol=1e-6, atol=1e-6)


def test_standalone_measure_functions_match_decomposition() -> None:
    sample = _categorical_sample()

    d_ord_ent = OrdinalEntropyDecomposition(sample)
    assert jnp.allclose(ordinal_entropy_of_expected_predictive_distribution(sample), d_ord_ent.total)
    assert jnp.allclose(ordinal_conditional_entropy(sample), d_ord_ent.aleatoric)

    d_ord_var = OrdinalVarianceDecomposition(sample)
    assert jnp.allclose(ordinal_variance_of_expected_predictive_distribution(sample), d_ord_var.total)
    assert jnp.allclose(ordinal_conditional_variance(sample), d_ord_var.aleatoric)

    d_lw_ent = LabelwiseBinaryEntropyDecomposition(sample)
    assert jnp.allclose(labelwise_entropy_of_expected_predictive_distribution(sample), d_lw_ent.total)
    assert jnp.allclose(labelwise_conditional_entropy(sample), d_lw_ent.aleatoric)

    d_lw_var = LabelwiseBinaryVarianceDecomposition(sample)
    assert jnp.allclose(labelwise_variance_of_expected_predictive_distribution(sample), d_lw_var.total)
    assert jnp.allclose(labelwise_conditional_variance(sample), d_lw_var.aleatoric)

    d_cat_var = CategoricalVarianceDecomposition(sample)
    assert jnp.allclose(categorical_variance_total(sample), d_cat_var.total)
    assert jnp.allclose(categorical_variance_aleatoric(sample), d_cat_var.aleatoric)


def test_gaussian_variance_decomposition_values() -> None:
    sample = _gaussian_sample()
    d = SecondOrderVarianceDecomposition(sample)

    expected_au = jnp.array([0.5])
    expected_eu = jnp.var(jnp.array([1.0, 2.0, 3.0]), ddof=0)

    assert jnp.allclose(d.aleatoric, expected_au, rtol=1e-6)
    assert jnp.allclose(d.epistemic, expected_eu, rtol=1e-6)
    assert jnp.allclose(d.total, expected_au + expected_eu, rtol=1e-6)


def test_gaussian_variance_decomposition_is_additive() -> None:
    sample = _gaussian_sample()
    d = SecondOrderVarianceDecomposition(sample)
    assert jnp.allclose(d.total, d.aleatoric + d.epistemic, rtol=1e-6, atol=1e-6)


def test_identical_gaussian_has_zero_epistemic() -> None:
    sample = _identical_gaussian_sample()
    d = SecondOrderVarianceDecomposition(sample)
    assert jnp.allclose(d.epistemic, 0.0, atol=1e-6)


def test_gaussian_standalone_functions_match_decomposition() -> None:
    sample = _gaussian_sample()
    d = SecondOrderVarianceDecomposition(sample)
    assert jnp.allclose(conditional_variance(sample), d.aleatoric)
    assert jnp.allclose(mutual_information_variance(sample), d.epistemic)


def test_ordinal_entropy_with_log_base() -> None:
    sample = _categorical_sample()
    d_nats = OrdinalEntropyDecomposition(sample, base=None)
    d_bits = OrdinalEntropyDecomposition(sample, base=2)
    d_norm = OrdinalEntropyDecomposition(sample, base="normalize")

    assert jnp.allclose(d_bits.total, d_nats.total / jnp.log(2), rtol=1e-6)
    assert jnp.allclose(d_norm.total, d_nats.total / jnp.log(2), rtol=1e-6)


def _bh(x: jnp.ndarray) -> jnp.ndarray:
    return scipy_entropy(jnp.stack([x, 1.0 - x], axis=-1), axis=-1)


def test_labelwise_entropy_vs_manual_formula() -> None:
    sample = _categorical_sample()
    p = sample.array.probabilities  # (M=3, N=2, K=3)
    axis = sample.sample_axis
    p_bar = jnp.mean(p, axis=axis)  # (N=2, K=3)

    expected_tu = jnp.sum(_bh(p_bar), axis=-1)
    expected_au = jnp.mean(jnp.sum(_bh(p), axis=-1), axis=axis)

    d = LabelwiseBinaryEntropyDecomposition(sample)
    assert jnp.allclose(d.total, expected_tu, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(d.aleatoric, expected_au, rtol=1e-6, atol=1e-6)


def test_labelwise_variance_vs_manual_formula() -> None:
    sample = _categorical_sample()
    p = sample.array.probabilities  # (M, N, K)
    axis = sample.sample_axis
    p_bar = jnp.mean(p, axis=axis)  # (N, K)

    expected_tu = jnp.sum(p_bar * (1.0 - p_bar), axis=-1)
    expected_au = jnp.mean(jnp.sum(p * (1.0 - p), axis=-1), axis=axis)

    d = LabelwiseBinaryVarianceDecomposition(sample)
    assert jnp.allclose(d.total, expected_tu, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(d.aleatoric, expected_au, rtol=1e-6, atol=1e-6)


def test_labelwise_single_distribution_measures() -> None:
    probs = jnp.array([[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]], dtype=float)
    dist = JaxProbabilityCategoricalDistribution(probs)
    p = dist.probabilities  # (N=2, K=3)

    assert jnp.allclose(labelwise_entropy(dist), jnp.sum(_bh(p), axis=-1), rtol=1e-6)
    assert jnp.allclose(labelwise_variance(dist), jnp.sum(p * (1.0 - p), axis=-1), rtol=1e-6)


def test_labelwise_entropy_with_log_base() -> None:
    sample = _categorical_sample()
    d_nats = LabelwiseBinaryEntropyDecomposition(sample, base=None)
    d_bits = LabelwiseBinaryEntropyDecomposition(sample, base=2)
    d_norm = LabelwiseBinaryEntropyDecomposition(sample, base="normalize")

    assert jnp.allclose(d_bits.total, d_nats.total / jnp.log(2), rtol=1e-6)
    assert jnp.allclose(d_norm.total, d_bits.total, rtol=1e-6)
