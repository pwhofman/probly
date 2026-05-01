"""Tests for entropy decomposition on NumPy representations."""

from __future__ import annotations

import numpy as np

from probly.quantification import SecondOrderEntropyDecomposition, quantify
from probly.quantification.measure.distribution import (
    conditional_entropy,
    entropy_of_expected_predictive_distribution,
    mutual_information,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution


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
        array=ArrayCategoricalDistribution(probabilities),
        sample_axis=0,
    )


def test_quantify_dispatches_to_entropy_decomposition_for_array_second_order_distribution() -> None:
    distribution = _array_dirichlet_distribution()

    decomposition = quantify(distribution)

    assert isinstance(decomposition, SecondOrderEntropyDecomposition)


def test_array_second_order_distribution_decomposition_matches_measure_functions() -> None:
    distribution = _array_dirichlet_distribution()

    decomposition = quantify(distribution)

    np.testing.assert_allclose(
        decomposition.total, entropy_of_expected_predictive_distribution(distribution), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(decomposition.aleatoric, conditional_entropy(distribution), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decomposition.epistemic, mutual_information(distribution), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        decomposition.total, decomposition.aleatoric + decomposition.epistemic, rtol=1e-12, atol=1e-12
    )


def test_quantify_dispatches_to_entropy_decomposition_for_array_distribution_sample() -> None:
    sample = _array_categorical_sample()

    decomposition = quantify(sample)

    assert isinstance(decomposition, SecondOrderEntropyDecomposition)


def test_array_distribution_sample_decomposition_matches_measure_functions() -> None:
    sample = _array_categorical_sample()

    decomposition = quantify(sample)

    np.testing.assert_allclose(
        decomposition.total, entropy_of_expected_predictive_distribution(sample), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(decomposition.aleatoric, conditional_entropy(sample), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decomposition.epistemic, mutual_information(sample), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        decomposition.total, decomposition.aleatoric + decomposition.epistemic, rtol=1e-12, atol=1e-12
    )


def test_array_second_order_distribution_notion_access_and_types_match_backend() -> None:
    decomposition = quantify(_array_dirichlet_distribution())

    total = decomposition["tu"]
    aleatoric = decomposition["au"]
    epistemic = decomposition["eu"]

    assert isinstance(total, np.ndarray)
    assert isinstance(aleatoric, np.ndarray)
    assert isinstance(epistemic, np.ndarray)


def test_array_decomposition_notion_access_and_types_match_backend() -> None:
    decomposition = quantify(_array_categorical_sample())

    total = decomposition["tu"]
    aleatoric = decomposition["au"]
    epistemic = decomposition["eu"]

    assert isinstance(total, np.ndarray)
    assert isinstance(aleatoric, np.ndarray)
    assert isinstance(epistemic, np.ndarray)


def test_array_decomposition_caches_component_objects() -> None:
    decomposition = quantify(_array_categorical_sample())

    total = decomposition.total
    aleatoric = decomposition.aleatoric
    epistemic = decomposition.epistemic

    assert decomposition.total is total
    assert decomposition.aleatoric is aleatoric
    assert decomposition.epistemic is epistemic
    assert decomposition["tu"] is total
    assert decomposition["au"] is aleatoric
    assert decomposition["eu"] is epistemic
