"""Tests for zero-one decomposition on NumPy representations."""

from __future__ import annotations

import numpy as np

from probly.quantification import SecondOrderZeroOneDecomposition
from probly.quantification.measure.distribution import (
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
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
        array=ArrayCategoricalDistribution(probabilities),
        sample_axis=0,
    )


def test_array_zero_one_decomposition_matches_measure_functions() -> None:
    sample = _array_categorical_sample()

    decomposition = SecondOrderZeroOneDecomposition(sample)

    np.testing.assert_allclose(
        decomposition.total, max_probability_complement_of_expected(sample), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        decomposition.aleatoric, expected_max_probability_complement(sample), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(decomposition.epistemic, max_disagreement(sample), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        decomposition.total, decomposition.aleatoric + decomposition.epistemic, rtol=1e-12, atol=1e-12
    )


def test_array_zero_one_decomposition_known_values() -> None:
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

    decomposition = SecondOrderZeroOneDecomposition(sample)

    np.testing.assert_allclose(decomposition.total, 0.45, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decomposition.aleatoric, 0.15, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decomposition.epistemic, 0.30, rtol=1e-12, atol=1e-12)


def test_array_zero_one_decomposition_notion_access_and_types_match_backend() -> None:
    decomposition = SecondOrderZeroOneDecomposition(_array_categorical_sample())

    total = decomposition["tu"]
    aleatoric = decomposition["au"]
    epistemic = decomposition["eu"]

    assert isinstance(total, np.ndarray)
    assert isinstance(aleatoric, np.ndarray)
    assert isinstance(epistemic, np.ndarray)


def test_array_zero_one_decomposition_caches_component_objects() -> None:
    decomposition = SecondOrderZeroOneDecomposition(_array_categorical_sample())

    total = decomposition.total
    aleatoric = decomposition.aleatoric
    epistemic = decomposition.epistemic

    assert decomposition.total is total
    assert decomposition.aleatoric is aleatoric
    assert decomposition.epistemic is epistemic
    assert decomposition["tu"] is total
    assert decomposition["au"] is aleatoric
    assert decomposition["eu"] is epistemic
