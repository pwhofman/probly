"""Tests for NumPy/SciPy distribution measures."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from scipy.stats import dirichlet, entropy as scipy_entropy, norm

from probly.quantification.measure.distribution import (
    conditional_entropy,
    dempster_shafer_uncertainty,
    entropy,
    entropy_of_expected_predictive_distribution,
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
    mutual_information,
    vacuity,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution
from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution

CATEGORICAL_BASES: tuple[None | float | Literal["normalize"], ...] = (None, 2.0, "normalize")
NUMERIC_BASES: tuple[None | float, ...] = (None, 2.0, 10.0)


def _resolve_categorical_base(base: None | float | Literal["normalize"], num_classes: int) -> None | float:
    if base == "normalize":
        return float(num_classes)
    return base


def _change_base_natural_log(values: np.ndarray, base: None | float) -> np.ndarray:
    if base is None or base == np.e:
        return values
    return values / np.log(base)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([[0.25, 0.25, 0.5]], dtype=float),
        np.array([[0.1, 0.2, 0.7], [0.4, 0.1, 0.5]], dtype=float),
    ],
)
@pytest.mark.parametrize("base", CATEGORICAL_BASES)
def test_array_categorical_entropy_matches_scipy(
    probabilities: np.ndarray, base: None | float | Literal["normalize"]
) -> None:
    distribution = ArrayCategoricalDistribution(probabilities)

    measured = entropy(distribution, base=base)
    expected = scipy_entropy(probabilities, axis=-1, base=_resolve_categorical_base(base, probabilities.shape[-1]))

    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_array_categorical_entropy_normalize_maps_to_unit_interval() -> None:
    probabilities = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    measured = entropy(ArrayCategoricalDistribution(probabilities), base="normalize")

    np.testing.assert_allclose(measured[0], 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(measured[1], 0.0, rtol=1e-12, atol=1e-12)
    assert np.all(measured >= 0.0)
    assert np.all(measured <= 1.0)


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_array_dirichlet_entropy_matches_scipy(base: None | float) -> None:
    alphas = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5],
        ],
        dtype=float,
    )
    distribution = ArrayDirichletDistribution(alphas)

    measured = entropy(distribution, base=base)
    expected_natural = np.array([dirichlet(alpha).entropy() for alpha in alphas], dtype=float)
    expected = _change_base_natural_log(expected_natural, base)

    np.testing.assert_allclose(measured, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_array_gaussian_entropy_matches_scipy_norm(base: None | float) -> None:
    mean = np.array([0.0, 3.5, -1.0], dtype=float)
    var = np.array([1.0, 0.25, 2.0], dtype=float)
    distribution = ArrayGaussianDistribution(mean=mean, var=var)

    measured = entropy(distribution, base=base)
    expected_natural = norm(scale=np.sqrt(var)).entropy()
    expected = _change_base_natural_log(expected_natural, base)

    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_array_sample_second_order_measures_match_scipy(
    sample_axis: int, base: None | float | Literal["normalize"]
) -> None:
    base_probabilities = np.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    probabilities = np.moveaxis(base_probabilities, 0, sample_axis)
    sample = ArrayCategoricalDistributionSample(
        array=ArrayCategoricalDistribution(probabilities),
        sample_axis=sample_axis,
    )

    measured_entropy_of_expected = entropy_of_expected_predictive_distribution(sample, base=base)
    measured_conditional_entropy = conditional_entropy(sample, base=base)
    measured_mutual_information = mutual_information(sample, base=base)

    expected_mean = np.mean(probabilities, axis=sample_axis)
    scipy_base = _resolve_categorical_base(base, probabilities.shape[-1])
    expected_entropy_of_expected = scipy_entropy(expected_mean, axis=-1, base=scipy_base)
    expected_conditional_entropy = np.mean(scipy_entropy(probabilities, axis=-1, base=scipy_base), axis=sample_axis)
    expected_mutual_information = np.mean(
        scipy_entropy(
            probabilities,
            np.expand_dims(expected_mean, axis=sample_axis),
            axis=-1,
            base=scipy_base,
        ),
        axis=sample_axis,
    )

    np.testing.assert_allclose(measured_entropy_of_expected, expected_entropy_of_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(measured_conditional_entropy, expected_conditional_entropy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(measured_mutual_information, expected_mutual_information, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
def test_array_dirichlet_entropy_of_expected_predictive_distribution_matches_scipy(
    base: None | float | Literal["normalize"],
) -> None:
    alphas = np.array(
        [
            [2.0, 3.0, 5.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    distribution = ArrayDirichletDistribution(alphas)

    measured = entropy_of_expected_predictive_distribution(distribution, base=base)
    expected_mean = alphas / np.sum(alphas, axis=-1, keepdims=True)
    expected = scipy_entropy(expected_mean, axis=-1, base=_resolve_categorical_base(base, expected_mean.shape[-1]))

    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_array_dirichlet_conditional_entropy_and_mutual_information_known_points(base: None | float) -> None:
    num_classes = 3
    expected_uniform_entropy = _change_base_natural_log(np.asarray(np.log(num_classes), dtype=float), base)

    concentrated = ArrayDirichletDistribution(np.array([1000.0, 1000.0, 1000.0], dtype=float))
    concentrated_conditional = conditional_entropy(concentrated, base=base)
    concentrated_mutual_information = mutual_information(concentrated, base=base)
    concentrated_entropy_of_expected = entropy_of_expected_predictive_distribution(concentrated, base=base)

    np.testing.assert_allclose(concentrated_entropy_of_expected, expected_uniform_entropy, atol=1e-12)
    assert concentrated_conditional == pytest.approx(expected_uniform_entropy, abs=2e-3)
    assert concentrated_mutual_information >= 0.0
    assert concentrated_mutual_information < 1e-3

    corner_like = ArrayDirichletDistribution(np.array([1e-3, 1e-3, 1e-3], dtype=float))
    corner_like_conditional = conditional_entropy(corner_like, base=base)
    corner_like_mutual_information = mutual_information(corner_like, base=base)
    corner_like_entropy_of_expected = entropy_of_expected_predictive_distribution(corner_like, base=base)

    np.testing.assert_allclose(corner_like_entropy_of_expected, expected_uniform_entropy, atol=1e-12)
    assert corner_like_conditional >= 0.0
    assert corner_like_conditional < 1e-2
    assert corner_like_mutual_information > 0.95 * expected_uniform_entropy
    assert corner_like_mutual_information <= corner_like_entropy_of_expected


def test_array_normalize_base_unsupported_for_non_categorical_entropies() -> None:
    dirichlet_distribution = ArrayDirichletDistribution(np.array([2.0, 3.0, 5.0], dtype=float))
    gaussian_distribution = ArrayGaussianDistribution(
        mean=np.array([0.0], dtype=float), var=np.array([1.0], dtype=float)
    )

    with pytest.raises(ValueError, match="normalization is not supported"):
        entropy(dirichlet_distribution, base="normalize")

    with pytest.raises(ValueError, match="normalization is not supported"):
        entropy(gaussian_distribution, base="normalize")

    with pytest.raises(ValueError, match="normalization is not supported"):
        conditional_entropy(dirichlet_distribution, base="normalize")

    with pytest.raises(ValueError, match="normalization is not supported"):
        mutual_information(dirichlet_distribution, base="normalize")


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_identity_holds_for_array_dirichlet(base: None | float) -> None:
    alphas = np.array(
        [
            [1.5, 2.0, 3.5],
            [10.0, 10.0, 10.0],
            [1e-2, 2e-2, 3e-2],
        ],
        dtype=float,
    )
    distribution = ArrayDirichletDistribution(alphas)

    expected_entropy = entropy_of_expected_predictive_distribution(distribution, base=base)
    decomposition_sum = conditional_entropy(distribution, base=base) + mutual_information(distribution, base=base)

    np.testing.assert_allclose(expected_entropy, decomposition_sum, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_identity_holds_for_array_categorical_sample(sample_axis: int, base: None | float | str) -> None:
    base_probabilities = np.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    probabilities = np.moveaxis(base_probabilities, 0, sample_axis)
    sample = ArrayCategoricalDistributionSample(
        array=ArrayCategoricalDistribution(probabilities),
        sample_axis=sample_axis,
    )

    expected_entropy = entropy_of_expected_predictive_distribution(sample, base=base)
    decomposition_sum = conditional_entropy(sample, base=base) + mutual_information(sample, base=base)

    np.testing.assert_allclose(expected_entropy, decomposition_sum, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_array_sample_zero_one_measures_match_manual(sample_axis: int) -> None:
    base_probabilities = np.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    probabilities = np.moveaxis(base_probabilities, 0, sample_axis)
    sample = ArrayCategoricalDistributionSample(
        array=ArrayCategoricalDistribution(probabilities),
        sample_axis=sample_axis,
    )

    measured_total = max_probability_complement_of_expected(sample)
    measured_aleatoric = expected_max_probability_complement(sample)
    measured_epistemic = max_disagreement(sample)

    expected_mean = np.mean(probabilities, axis=sample_axis)
    expected_total = 1.0 - np.max(expected_mean, axis=-1)
    expected_aleatoric = np.mean(1.0 - np.max(probabilities, axis=-1), axis=sample_axis)
    bma_argmax_expanded = np.expand_dims(np.argmax(expected_mean, axis=-1), axis=(sample_axis, -1))
    per_sample_bma_prob = np.take_along_axis(probabilities, bma_argmax_expanded, axis=-1).squeeze(-1)
    expected_epistemic = np.mean(np.max(probabilities, axis=-1) - per_sample_bma_prob, axis=sample_axis)

    np.testing.assert_allclose(measured_total, expected_total, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(measured_aleatoric, expected_aleatoric, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(measured_epistemic, expected_epistemic, rtol=1e-12, atol=1e-12)


def test_array_sample_zero_one_known_values() -> None:
    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
        ],
        dtype=float,
    )
    sample = ArrayCategoricalDistributionSample(
        array=ArrayCategoricalDistribution(probabilities),
        sample_axis=0,
    )

    np.testing.assert_allclose(max_probability_complement_of_expected(sample), 0.45, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(expected_max_probability_complement(sample), 0.15, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(max_disagreement(sample), 0.30, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_zero_one_identity_holds_for_array_categorical_sample(sample_axis: int) -> None:
    base_probabilities = np.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    probabilities = np.moveaxis(base_probabilities, 0, sample_axis)
    sample = ArrayCategoricalDistributionSample(
        array=ArrayCategoricalDistribution(probabilities),
        sample_axis=sample_axis,
    )

    total = max_probability_complement_of_expected(sample)
    aleatoric = expected_max_probability_complement(sample)
    epistemic = max_disagreement(sample)

    np.testing.assert_allclose(total, aleatoric + epistemic, rtol=1e-12, atol=1e-12)


def test_array_dirichlet_vacuity_known_values() -> None:
    alphas = np.array(
        [
            [1.0, 1.0, 1.0],  # uniform Dir(1,1,1): K=3, alpha_0=3 -> vacuity=1
            [10.0, 10.0, 10.0],  # K=3, alpha_0=30 -> vacuity=0.1
            [2.0, 3.0, 5.0],  # K=3, alpha_0=10 -> vacuity=0.3
        ],
        dtype=float,
    )
    distribution = ArrayDirichletDistribution(alphas)

    measured = vacuity(distribution)

    np.testing.assert_allclose(measured, np.array([1.0, 0.1, 0.3]), rtol=1e-12, atol=1e-12)


def test_array_dirichlet_vacuity_lies_in_unit_interval() -> None:
    rng = np.random.default_rng(seed=0)
    alphas = rng.uniform(low=1.0, high=20.0, size=(50, 4))
    distribution = ArrayDirichletDistribution(alphas)

    measured = vacuity(distribution)

    assert np.all(measured > 0.0)
    assert np.all(measured <= 1.0)


def test_array_dirichlet_vacuity_decreases_with_evidence() -> None:
    weak = ArrayDirichletDistribution(np.array([1.0, 1.0, 1.0], dtype=float))
    strong = ArrayDirichletDistribution(np.array([100.0, 100.0, 100.0], dtype=float))

    assert vacuity(weak) > vacuity(strong)


def test_array_dirichlet_max_probability_complement_of_expected_known_values() -> None:
    alphas = np.array(
        [
            [1.0, 1.0, 1.0],  # uniform: max(1/3) -> 1 - 1/3 = 2/3
            [10.0, 1.0, 1.0],  # max = 10/12 -> 1 - 5/6 = 1/6
            [2.0, 3.0, 5.0],  # max = 5/10 -> 1 - 1/2 = 1/2
        ],
        dtype=float,
    )
    distribution = ArrayDirichletDistribution(alphas)

    measured = max_probability_complement_of_expected(distribution)

    expected = np.array([2.0 / 3.0, 1.0 / 6.0, 0.5])
    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_array_dirichlet_max_probability_complement_of_expected_matches_explicit_formula() -> None:
    rng = np.random.default_rng(seed=0)
    alphas = rng.uniform(low=0.5, high=20.0, size=(50, 5))
    distribution = ArrayDirichletDistribution(alphas)

    measured = max_probability_complement_of_expected(distribution)

    expected_mean = alphas / alphas.sum(axis=-1, keepdims=True)
    expected = 1.0 - np.max(expected_mean, axis=-1)
    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_array_dirichlet_max_probability_complement_of_expected_lies_in_unit_interval() -> None:
    rng = np.random.default_rng(seed=1)
    alphas = rng.uniform(low=0.1, high=50.0, size=(50, 6))
    distribution = ArrayDirichletDistribution(alphas)

    measured = max_probability_complement_of_expected(distribution)

    assert np.all(measured >= 0.0)
    assert np.all(measured < 1.0)


def test_array_dirichlet_max_probability_complement_of_expected_invariant_to_scaling() -> None:
    """Scaling the alphas by a constant leaves the predictive mean (and thus the score) unchanged."""
    base = np.array([1.0, 2.0, 3.0], dtype=float)
    weak = ArrayDirichletDistribution(base)
    strong = ArrayDirichletDistribution(100.0 * base)

    np.testing.assert_allclose(
        max_probability_complement_of_expected(weak),
        max_probability_complement_of_expected(strong),
        rtol=1e-12,
        atol=1e-12,
    )


def test_array_gaussian_dempster_shafer_uniform_logits_with_default_factor() -> None:
    """Uniform-zero logits should give vacuity = K / (K + K * exp(0)) = 1/2."""
    mean = np.zeros((3, 5), dtype=float)
    var = np.ones_like(mean)
    distribution = ArrayGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution)

    np.testing.assert_allclose(measured, 0.5, rtol=1e-12, atol=1e-12)


def test_array_gaussian_dempster_shafer_matches_explicit_formula() -> None:
    import math  # noqa: PLC0415

    rng = np.random.default_rng(seed=0)
    mean = rng.normal(loc=0.0, scale=2.0, size=(20, 5))
    var = rng.uniform(low=0.01, high=4.0, size=(20, 5))
    distribution = ArrayGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution)

    num_classes = mean.shape[-1]
    adjusted = mean / np.sqrt(1.0 + (math.pi / 8.0) * var)
    expected = num_classes / (num_classes + np.sum(np.exp(adjusted), axis=-1))
    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_array_gaussian_dempster_shafer_lies_in_unit_interval() -> None:
    rng = np.random.default_rng(seed=1)
    mean = rng.normal(loc=0.0, scale=5.0, size=(50, 4))
    var = rng.uniform(low=0.01, high=10.0, size=(50, 4))
    distribution = ArrayGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution)

    assert np.all(measured > 0.0)
    assert np.all(measured <= 1.0)


def test_array_gaussian_dempster_shafer_high_variance_increases_uncertainty() -> None:
    """Mean-field correction shrinks logits when variance is large -> vacuity goes up."""
    mean = np.array([[10.0, -10.0, 0.0, 0.0]], dtype=float)
    low_var = np.full_like(mean, 1e-3)
    high_var = np.full_like(mean, 1000.0)

    low_var_score = dempster_shafer_uncertainty(ArrayGaussianDistribution(mean=mean, var=low_var))
    high_var_score = dempster_shafer_uncertainty(ArrayGaussianDistribution(mean=mean, var=high_var))

    assert high_var_score[0] > low_var_score[0]


def test_array_gaussian_dempster_shafer_zero_factor_disables_mean_field() -> None:
    """``mean_field_factor=0`` should reduce to the variance-free formula K / (K + sum exp(h))."""
    mean = np.array([[1.0, 2.0, 3.0]], dtype=float)
    var = np.array([[100.0, 100.0, 100.0]], dtype=float)
    distribution = ArrayGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution, mean_field_factor=0.0)

    expected = 3.0 / (3.0 + np.sum(np.exp(mean), axis=-1))
    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)
