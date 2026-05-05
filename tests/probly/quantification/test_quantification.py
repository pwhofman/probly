"""Tests for top-level quantification dispatchers."""

from __future__ import annotations

import numpy as np

from probly.quantification import SecondOrderEntropyDecomposition, decompose, measure, quantify
from probly.quantification.decomposition.decomposition import ConstantTotalDecomposition
from probly.quantification.measure.distribution import entropy_of_expected_predictive_distribution
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistributionSample,
    ArrayProbabilityCategoricalDistribution,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution
from probly.representation.sample import ArraySample


def _array_sample() -> ArraySample[np.ndarray]:
    return ArraySample(
        array=np.array(
            [
                [1.0, 2.0],
                [3.0, 6.0],
                [5.0, 10.0],
            ],
            dtype=float,
        ),
        sample_axis=0,
    )


def _array_dirichlet_distribution() -> ArrayDirichletDistribution:
    return ArrayDirichletDistribution(
        np.array(
            [
                [2.0, 3.0, 5.0],
                [1.0, 4.0, 2.0],
            ],
            dtype=float,
        )
    )


def _array_categorical_sample() -> ArrayCategoricalDistributionSample:
    probabilities = np.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    return ArrayCategoricalDistributionSample(
        array=ArrayProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )


def test_measure_dispatches_to_registered_sample_measure() -> None:
    sample = _array_sample()

    uncertainty = measure(sample)

    np.testing.assert_allclose(uncertainty, sample.sample_var(), rtol=1e-12, atol=1e-12)


def test_decompose_wraps_registered_measure_as_constant_total_decomposition() -> None:
    sample = _array_sample()

    decomposition = decompose(sample)

    assert isinstance(decomposition, ConstantTotalDecomposition)
    np.testing.assert_allclose(decomposition.total, sample.sample_var(), rtol=1e-12, atol=1e-12)


def test_decompose_dispatches_to_registered_entropy_decomposition() -> None:
    decomposition = decompose(_array_categorical_sample())

    assert isinstance(decomposition, SecondOrderEntropyDecomposition)


def test_measure_falls_back_to_registered_decomposition_total() -> None:
    distribution = _array_dirichlet_distribution()

    uncertainty = measure(distribution)

    expected = entropy_of_expected_predictive_distribution(distribution)

    np.testing.assert_allclose(uncertainty, expected, rtol=1e-12, atol=1e-12)


def test_measure_prefers_registered_decomposition_for_distribution_sample() -> None:
    sample = _array_categorical_sample()

    uncertainty = measure(sample)

    expected = entropy_of_expected_predictive_distribution(sample)

    np.testing.assert_allclose(uncertainty, expected, rtol=1e-12, atol=1e-12)


def test_quantify_prefers_registered_decomposition_for_distribution_sample() -> None:
    quantification = quantify(_array_categorical_sample())

    assert isinstance(quantification, SecondOrderEntropyDecomposition)


def test_quantify_uses_decompose_fallback_for_plain_sample() -> None:
    sample = _array_sample()

    quantification = quantify(sample)

    assert isinstance(quantification, ConstantTotalDecomposition)
    np.testing.assert_allclose(quantification.total, sample.sample_var(), rtol=1e-12, atol=1e-12)
