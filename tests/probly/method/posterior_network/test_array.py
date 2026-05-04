"""Tests for ``PosteriorNetworkDecomposition`` on numpy Dirichlet representations."""

from __future__ import annotations

import numpy as np
import pytest

from probly.method.posterior_network import PosteriorNetworkDecomposition
from probly.quantification import AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty
from probly.quantification.measure.distribution import (
    entropy_of_expected_predictive_distribution,
    max_probability_complement_of_expected,
    vacuity,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution

NUMERIC_BASES: tuple[None | float, ...] = (None, 2.0, 10.0)


def _array_dirichlet() -> ArrayDirichletDistribution:
    alphas = np.array(
        [
            [2.0, 3.0, 5.0],  # alpha_0=10
            [1.0, 1.0, 1.0],  # uniform, alpha_0=3
            [10.0, 10.0, 10.0],  # alpha_0=30
            [100.0, 1.0, 1.0],  # peaked, alpha_0=102
        ],
        dtype=float,
    )
    return ArrayDirichletDistribution(alphas=alphas)


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_array_decomposition_components_match_measure_functions(base: None | float) -> None:
    distribution = _array_dirichlet()

    decomposition = PosteriorNetworkDecomposition(distribution, base=base)

    np.testing.assert_allclose(
        decomposition.total,
        max_probability_complement_of_expected(distribution),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        decomposition.aleatoric,
        entropy_of_expected_predictive_distribution(distribution, base=base),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(decomposition.epistemic, vacuity(distribution), rtol=1e-12, atol=1e-12)


def test_array_decomposition_notion_access() -> None:
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    np.testing.assert_array_equal(decomposition[TotalUncertainty], decomposition.total)
    np.testing.assert_array_equal(decomposition[AleatoricUncertainty], decomposition.aleatoric)
    np.testing.assert_array_equal(decomposition[EpistemicUncertainty], decomposition.epistemic)
    np.testing.assert_array_equal(decomposition["tu"], decomposition.total)
    np.testing.assert_array_equal(decomposition["au"], decomposition.aleatoric)
    np.testing.assert_array_equal(decomposition["eu"], decomposition.epistemic)


def test_array_decomposition_caches_components() -> None:
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    total = decomposition.total
    aleatoric = decomposition.aleatoric
    epistemic = decomposition.epistemic

    assert decomposition.total is total
    assert decomposition.aleatoric is aleatoric
    assert decomposition.epistemic is epistemic


def test_array_decomposition_returns_ndarrays() -> None:
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    assert isinstance(decomposition.total, np.ndarray)
    assert isinstance(decomposition.aleatoric, np.ndarray)
    assert isinstance(decomposition.epistemic, np.ndarray)


def test_array_decomposition_total_in_unit_interval() -> None:
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    assert np.all(decomposition.total >= 0.0)
    assert np.all(decomposition.total < 1.0)


def test_array_decomposition_canonical_notion_is_total() -> None:
    """The canonical notion of the PostNet decomposition is total (the misclassification-detection score)."""
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    np.testing.assert_array_equal(decomposition.get_canonical(), decomposition.total)


def test_array_decomposition_uniform_dirichlet_has_max_total_and_max_vacuity() -> None:
    """A uniform Dir(1,...,1) is the maximally-uncertain case for both TU and EU."""
    uniform = ArrayDirichletDistribution(alphas=np.array([1.0, 1.0, 1.0, 1.0], dtype=float))
    decomposition = PosteriorNetworkDecomposition(uniform)

    # TU = 1 - 1/K = 1 - 1/4 = 0.75
    np.testing.assert_allclose(decomposition.total, 0.75, rtol=1e-12, atol=1e-12)
    # EU = K/K = 1.0
    np.testing.assert_allclose(decomposition.epistemic, 1.0, rtol=1e-12, atol=1e-12)
