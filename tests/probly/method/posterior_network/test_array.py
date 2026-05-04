"""Tests for ``PosteriorNetworkDecomposition`` on numpy Dirichlet representations."""

from __future__ import annotations

import numpy as np
import pytest

from probly.method.posterior_network import PosteriorNetworkDecomposition
from probly.quantification import AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty
from probly.quantification.measure.distribution import (
    max_probability_complement_of_expected,
    vacuity,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution


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


def test_array_decomposition_components_match_measure_functions() -> None:
    distribution = _array_dirichlet()

    decomposition = PosteriorNetworkDecomposition(distribution)

    np.testing.assert_allclose(
        decomposition.aleatoric,
        max_probability_complement_of_expected(distribution),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(decomposition.epistemic, vacuity(distribution), rtol=1e-12, atol=1e-12)


def test_array_decomposition_components_only_aleatoric_and_epistemic() -> None:
    """Paper has no formal TU; decomposition has only AU and EU slots."""
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    assert decomposition.components == [AleatoricUncertainty, EpistemicUncertainty]
    assert len(decomposition) == 2


def test_array_decomposition_notion_access() -> None:
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    np.testing.assert_array_equal(decomposition[AleatoricUncertainty], decomposition.aleatoric)
    np.testing.assert_array_equal(decomposition[EpistemicUncertainty], decomposition.epistemic)
    np.testing.assert_array_equal(decomposition["au"], decomposition.aleatoric)
    np.testing.assert_array_equal(decomposition["eu"], decomposition.epistemic)


def test_array_decomposition_does_not_expose_total() -> None:
    """The paper has no formal total uncertainty measure; decomposition reflects that."""
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    with pytest.raises(KeyError):
        decomposition[TotalUncertainty]
    with pytest.raises(KeyError):
        decomposition["tu"]


def test_array_decomposition_caches_components() -> None:
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    aleatoric = decomposition.aleatoric
    epistemic = decomposition.epistemic

    assert decomposition.aleatoric is aleatoric
    assert decomposition.epistemic is epistemic


def test_array_decomposition_returns_ndarrays() -> None:
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    assert isinstance(decomposition.aleatoric, np.ndarray)
    assert isinstance(decomposition.epistemic, np.ndarray)


def test_array_decomposition_aleatoric_is_max_prob_complement() -> None:
    """AU = 1 - max_c (alpha_c / alpha_0): paper's Alea Conf complement (Tables 1-7)."""
    distribution = _array_dirichlet()
    decomposition = PosteriorNetworkDecomposition(distribution)

    expected = np.array(
        [
            1.0 - 5.0 / 10.0,  # max(2/10, 3/10, 5/10) = 5/10 -> 1 - 0.5 = 0.5
            1.0 - 1.0 / 3.0,  # uniform -> 1 - 1/3 = 2/3
            1.0 - 10.0 / 30.0,  # 1 - 1/3 = 2/3
            1.0 - 100.0 / 102.0,  # 1 - 100/102 = 2/102
        ]
    )
    np.testing.assert_allclose(decomposition.aleatoric, expected, rtol=1e-12, atol=1e-12)


def test_array_decomposition_aleatoric_in_unit_interval() -> None:
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    assert np.all(decomposition.aleatoric >= 0.0)
    assert np.all(decomposition.aleatoric < 1.0)


def test_array_decomposition_uniform_dirichlet_has_max_uncertainties() -> None:
    """A uniform Dir(1,...,1) is the maximally-uncertain case for both AU and EU."""
    uniform = ArrayDirichletDistribution(alphas=np.array([1.0, 1.0, 1.0, 1.0], dtype=float))
    decomposition = PosteriorNetworkDecomposition(uniform)

    # AU = 1 - 1/K = 1 - 1/4 = 0.75
    np.testing.assert_allclose(decomposition.aleatoric, 0.75, rtol=1e-12, atol=1e-12)
    # EU = K/K = 1.0
    np.testing.assert_allclose(decomposition.epistemic, 1.0, rtol=1e-12, atol=1e-12)


def test_array_decomposition_no_canonical_notion() -> None:
    """AU and EU are equally valid; no canonical notion."""
    decomposition = PosteriorNetworkDecomposition(_array_dirichlet())

    assert decomposition.canonical_notion is None
    with pytest.raises(NotImplementedError):
        decomposition.get_canonical()
