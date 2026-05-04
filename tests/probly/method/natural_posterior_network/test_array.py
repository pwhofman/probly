"""Tests for ``NaturalPosteriorDecomposition`` on numpy Dirichlet representations."""

from __future__ import annotations

import numpy as np
import pytest

from probly.method.natural_posterior_network import NaturalPosteriorDecomposition
from probly.quantification import AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty
from probly.quantification.measure.distribution import (
    entropy,
    entropy_of_expected_predictive_distribution,
    vacuity,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution

NUMERIC_BASES: tuple[None | float, ...] = (None, 2.0, 10.0)


def _array_dirichlet() -> ArrayDirichletDistribution:
    alphas = np.array(
        [
            [2.0, 3.0, 5.0],  # alpha_0=10, vacuity=0.3
            [1.0, 1.0, 1.0],  # alpha_0=3,  vacuity=1.0
            [10.0, 10.0, 10.0],  # alpha_0=30, vacuity=0.1
        ],
        dtype=float,
    )
    return ArrayDirichletDistribution(alphas=alphas)


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_array_decomposition_components_match_measure_functions(base: None | float) -> None:
    distribution = _array_dirichlet()

    decomposition = NaturalPosteriorDecomposition(distribution, base=base)

    np.testing.assert_allclose(
        decomposition.total, entropy(distribution, base=base), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        decomposition.aleatoric,
        entropy_of_expected_predictive_distribution(distribution, base=base),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(decomposition.epistemic, vacuity(distribution), rtol=1e-12, atol=1e-12)


def test_array_decomposition_notion_access() -> None:
    decomposition = NaturalPosteriorDecomposition(_array_dirichlet())

    np.testing.assert_array_equal(decomposition[TotalUncertainty], decomposition.total)
    np.testing.assert_array_equal(decomposition[AleatoricUncertainty], decomposition.aleatoric)
    np.testing.assert_array_equal(decomposition[EpistemicUncertainty], decomposition.epistemic)
    np.testing.assert_array_equal(decomposition["tu"], decomposition.total)
    np.testing.assert_array_equal(decomposition["au"], decomposition.aleatoric)
    np.testing.assert_array_equal(decomposition["eu"], decomposition.epistemic)


def test_array_decomposition_caches_components() -> None:
    decomposition = NaturalPosteriorDecomposition(_array_dirichlet())

    total = decomposition.total
    aleatoric = decomposition.aleatoric
    epistemic = decomposition.epistemic

    assert decomposition.total is total
    assert decomposition.aleatoric is aleatoric
    assert decomposition.epistemic is epistemic


def test_array_decomposition_returns_ndarrays() -> None:
    decomposition = NaturalPosteriorDecomposition(_array_dirichlet())

    assert isinstance(decomposition.total, np.ndarray)
    assert isinstance(decomposition.aleatoric, np.ndarray)
    assert isinstance(decomposition.epistemic, np.ndarray)


def test_array_decomposition_uniform_dirichlet_has_max_vacuity() -> None:
    uniform = ArrayDirichletDistribution(alphas=np.array([1.0, 1.0, 1.0], dtype=float))
    decomposition = NaturalPosteriorDecomposition(uniform)

    np.testing.assert_allclose(decomposition.epistemic, 1.0, rtol=1e-12, atol=1e-12)


def test_array_decomposition_canonical_notion_is_total() -> None:
    """The canonical notion of the NatPN decomposition is total uncertainty."""
    decomposition = NaturalPosteriorDecomposition(_array_dirichlet())

    np.testing.assert_array_equal(decomposition.get_canonical(), decomposition.total)
