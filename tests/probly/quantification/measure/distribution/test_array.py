"""Tests for NumPy/SciPy distribution measures."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import dirichlet, entropy as scipy_entropy, norm

from probly.quantification.measure.distribution import (
    conditional_entropy,
    entropy,
    entropy_of_expected_value,
    mutual_information,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution
from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([[0.25, 0.25, 0.5]], dtype=float),
        np.array([[0.1, 0.2, 0.7], [0.4, 0.1, 0.5]], dtype=float),
    ],
)
def test_array_categorical_entropy_matches_scipy(probabilities: np.ndarray) -> None:
    distribution = ArrayCategoricalDistribution(probabilities)

    measured = entropy(distribution)
    expected = scipy_entropy(probabilities, axis=-1)

    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_array_dirichlet_entropy_matches_scipy() -> None:
    alphas = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5],
        ],
        dtype=float,
    )
    distribution = ArrayDirichletDistribution(alphas)

    measured = entropy(distribution)
    expected = np.array([dirichlet(alpha).entropy() for alpha in alphas], dtype=float)

    np.testing.assert_allclose(measured, expected, rtol=1e-10, atol=1e-12)


def test_array_gaussian_entropy_matches_scipy_norm() -> None:
    mean = np.array([0.0, 3.5, -1.0], dtype=float)
    var = np.array([1.0, 0.25, 2.0], dtype=float)
    distribution = ArrayGaussianDistribution(mean=mean, var=var)

    measured = entropy(distribution)
    expected = norm(scale=np.sqrt(var)).entropy()

    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_array_sample_second_order_measures_match_scipy(sample_axis: int) -> None:
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

    measured_entropy_of_expected = entropy_of_expected_value(sample)
    measured_conditional_entropy = conditional_entropy(sample)
    measured_mutual_information = mutual_information(sample)

    expected_mean = np.mean(probabilities, axis=sample_axis)
    expected_entropy_of_expected = scipy_entropy(expected_mean, axis=-1)
    expected_conditional_entropy = np.mean(scipy_entropy(probabilities, axis=-1), axis=sample_axis)
    expected_mutual_information = np.mean(
        scipy_entropy(probabilities, np.expand_dims(expected_mean, axis=sample_axis), axis=-1),
        axis=sample_axis,
    )

    np.testing.assert_allclose(measured_entropy_of_expected, expected_entropy_of_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(measured_conditional_entropy, expected_conditional_entropy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(measured_mutual_information, expected_mutual_information, rtol=1e-12, atol=1e-12)


def test_array_dirichlet_entropy_of_expected_value_matches_scipy() -> None:
    alphas = np.array(
        [
            [2.0, 3.0, 5.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    distribution = ArrayDirichletDistribution(alphas)

    measured = entropy_of_expected_value(distribution)
    expected_mean = alphas / np.sum(alphas, axis=-1, keepdims=True)
    expected = scipy_entropy(expected_mean, axis=-1)

    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_array_dirichlet_conditional_entropy_and_mutual_information_known_points() -> None:
    num_classes = 3
    expected_uniform_entropy = np.log(num_classes)

    concentrated = ArrayDirichletDistribution(np.array([1000.0, 1000.0, 1000.0], dtype=float))
    concentrated_conditional = conditional_entropy(concentrated)
    concentrated_mutual_information = mutual_information(concentrated)
    concentrated_entropy_of_expected = entropy_of_expected_value(concentrated)

    np.testing.assert_allclose(concentrated_entropy_of_expected, expected_uniform_entropy, atol=1e-12)
    assert concentrated_conditional == pytest.approx(expected_uniform_entropy, abs=1e-3)
    assert concentrated_mutual_information >= 0.0
    assert concentrated_mutual_information < 1e-3

    corner_like = ArrayDirichletDistribution(np.array([1e-3, 1e-3, 1e-3], dtype=float))
    corner_like_conditional = conditional_entropy(corner_like)
    corner_like_mutual_information = mutual_information(corner_like)
    corner_like_entropy_of_expected = entropy_of_expected_value(corner_like)

    np.testing.assert_allclose(corner_like_entropy_of_expected, expected_uniform_entropy, atol=1e-12)
    assert corner_like_conditional >= 0.0
    assert corner_like_conditional < 1e-2
    assert corner_like_mutual_information > 0.95 * expected_uniform_entropy
    assert corner_like_mutual_information <= corner_like_entropy_of_expected


def test_identity_holds_for_array_dirichlet() -> None:
    alphas = np.array(
        [
            [1.5, 2.0, 3.5],
            [10.0, 10.0, 10.0],
            [1e-2, 2e-2, 3e-2],
        ],
        dtype=float,
    )
    distribution = ArrayDirichletDistribution(alphas)

    expected_entropy = entropy_of_expected_value(distribution)
    decomposition_sum = conditional_entropy(distribution) + mutual_information(distribution)

    np.testing.assert_allclose(expected_entropy, decomposition_sum, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_identity_holds_for_array_categorical_sample(sample_axis: int) -> None:
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

    expected_entropy = entropy_of_expected_value(sample)
    decomposition_sum = conditional_entropy(sample) + mutual_information(sample)

    np.testing.assert_allclose(expected_entropy, decomposition_sum, rtol=1e-12, atol=1e-12)
