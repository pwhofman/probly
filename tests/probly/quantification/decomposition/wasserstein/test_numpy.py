"""Tests for the distance-based (Wasserstein) decomposition on NumPy representations."""

from __future__ import annotations

import numpy as np

from probly.quantification import SecondOrderWassersteinDecomposition
from probly.quantification.measure.distribution import (
    expected_max_probability_complement,
    max_probability_complement_of_expected,
    min_expected_total_variation,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistributionSample,
    ArrayProbabilityCategoricalDistribution,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution


def _binary_sample() -> ArrayCategoricalDistributionSample:
    probabilities = np.array([[0.90, 0.10], [0.50, 0.50]], dtype=float)
    return ArrayCategoricalDistributionSample(
        array=ArrayProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )


def _batched_sample() -> ArrayCategoricalDistributionSample:
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


def test_array_wasserstein_decomposition_known_values_and_non_additive() -> None:
    decomposition = SecondOrderWassersteinDecomposition(_binary_sample())

    np.testing.assert_allclose(decomposition.total, 0.3, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(decomposition.aleatoric, 0.3, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(decomposition.epistemic, 0.2, rtol=1e-9, atol=1e-9)
    # The distance-based decomposition is not additive: TU != AU + EU.
    assert not np.allclose(decomposition.total, decomposition.aleatoric + decomposition.epistemic)


def test_array_wasserstein_decomposition_matches_measure_functions() -> None:
    sample = _binary_sample()
    decomposition = SecondOrderWassersteinDecomposition(sample)

    np.testing.assert_allclose(
        decomposition.total, max_probability_complement_of_expected(sample), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        decomposition.aleatoric, expected_max_probability_complement(sample), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(decomposition.epistemic, min_expected_total_variation(sample), rtol=1e-12, atol=1e-12)


def test_array_wasserstein_decomposition_satisfies_axiom_a3_and_ranges() -> None:
    """For a sample, AU <= TU and EU <= TU hold exactly, and TU <= (K-1)/K."""
    rng = np.random.default_rng(seed=0)
    logits = rng.normal(size=(20, 8, 4))  # (batch, samples, classes)
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    sample = ArrayCategoricalDistributionSample(
        array=ArrayProbabilityCategoricalDistribution(probabilities),
        sample_axis=1,
    )
    decomposition = SecondOrderWassersteinDecomposition(sample)
    total, aleatoric, epistemic = decomposition.total, decomposition.aleatoric, decomposition.epistemic

    assert np.all(aleatoric <= total + 1e-9)
    assert np.all(epistemic <= total + 1e-9)
    assert np.all(total <= 3.0 / 4.0 + 1e-9)
    assert np.all(aleatoric >= -1e-9)
    assert np.all(epistemic >= -1e-9)


def test_array_wasserstein_decomposition_notion_access() -> None:
    decomposition = SecondOrderWassersteinDecomposition(_batched_sample())

    assert isinstance(decomposition["tu"], np.ndarray)
    assert isinstance(decomposition["au"], np.ndarray)
    assert isinstance(decomposition["eu"], np.ndarray)


def test_array_wasserstein_decomposition_caches_components() -> None:
    decomposition = SecondOrderWassersteinDecomposition(_binary_sample())

    assert decomposition.total is decomposition.total
    assert decomposition.aleatoric is decomposition.aleatoric
    assert decomposition.epistemic is decomposition.epistemic


def test_array_wasserstein_decomposition_dirichlet_total_is_closed_form() -> None:
    distribution = ArrayDirichletDistribution(np.array([[2.0, 3.0, 5.0]], dtype=float))
    decomposition = SecondOrderWassersteinDecomposition(
        distribution, num_samples=2000, generator=np.random.default_rng(0)
    )

    # Total uncertainty has the closed form 1 - max_k alpha_k / alpha_0 = 1 - 5/10.
    np.testing.assert_allclose(decomposition.total, 0.5, rtol=1e-12, atol=1e-12)
    assert np.all(decomposition.aleatoric >= 0.0)
    assert np.all(decomposition.epistemic >= 0.0)
    assert np.all(decomposition.aleatoric <= 2.0 / 3.0 + 1e-9)
    assert np.all(decomposition.epistemic <= 2.0 / 3.0 + 1e-9)
