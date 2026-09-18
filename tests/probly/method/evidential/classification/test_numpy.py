"""Tests for ``EvidentialClassificationDecomposition`` on numpy Dirichlet representations."""

from __future__ import annotations

import numpy as np
import pytest

from probly.method.evidential.classification import EvidentialClassificationDecomposition
from probly.quantification import AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty
from probly.quantification.measure.distribution import vacuity
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution


def _array_dirichlet() -> ArrayDirichletDistribution:
    alphas = np.array(
        [
            [1.0, 1.0, 1.0],  # uniform: K=3, S=3, vacuity=1.0
            [10.0, 10.0, 10.0],  # K=3, S=30, vacuity=0.1
            [2.0, 3.0, 5.0],  # K=3, S=10, vacuity=0.3
            [100.0, 1.0, 1.0],  # K=3, S=102, vacuity=3/102
        ],
        dtype=float,
    )
    return ArrayDirichletDistribution(alphas=alphas)


def test_array_decomposition_epistemic_matches_vacuity() -> None:
    distribution = _array_dirichlet()

    decomposition = EvidentialClassificationDecomposition(distribution)

    np.testing.assert_allclose(decomposition.epistemic, vacuity(distribution), rtol=1e-12, atol=1e-12)


def test_array_decomposition_epistemic_known_values() -> None:
    distribution = _array_dirichlet()

    decomposition = EvidentialClassificationDecomposition(distribution)

    expected = np.array([1.0, 0.1, 0.3, 3.0 / 102.0])
    np.testing.assert_allclose(decomposition.epistemic, expected, rtol=1e-12, atol=1e-12)


def test_array_decomposition_components_only_epistemic() -> None:
    decomposition = EvidentialClassificationDecomposition(_array_dirichlet())

    assert decomposition.components == [EpistemicUncertainty]
    assert len(decomposition) == 1
    assert list(decomposition) == [EpistemicUncertainty]


def test_array_decomposition_canonical_notion_is_epistemic() -> None:
    decomposition = EvidentialClassificationDecomposition(_array_dirichlet())

    assert decomposition.canonical_notion is EpistemicUncertainty
    np.testing.assert_array_equal(decomposition.get_canonical(), decomposition.epistemic)


def test_array_decomposition_notion_access() -> None:
    decomposition = EvidentialClassificationDecomposition(_array_dirichlet())

    np.testing.assert_array_equal(decomposition[EpistemicUncertainty], decomposition.epistemic)
    np.testing.assert_array_equal(decomposition["eu"], decomposition.epistemic)
    np.testing.assert_array_equal(decomposition["epistemic"], decomposition.epistemic)


def test_array_decomposition_does_not_expose_aleatoric_or_total() -> None:
    """The paper has no aleatoric / total measures; the decomposition must reflect that."""
    decomposition = EvidentialClassificationDecomposition(_array_dirichlet())

    with pytest.raises(KeyError):
        decomposition[AleatoricUncertainty]
    with pytest.raises(KeyError):
        decomposition[TotalUncertainty]
    with pytest.raises(KeyError):
        decomposition["au"]
    with pytest.raises(KeyError):
        decomposition["tu"]


def test_array_decomposition_caches_component() -> None:
    decomposition = EvidentialClassificationDecomposition(_array_dirichlet())

    epistemic = decomposition.epistemic

    assert decomposition.epistemic is epistemic
    assert decomposition[EpistemicUncertainty] is epistemic


def test_array_decomposition_returns_ndarray() -> None:
    decomposition = EvidentialClassificationDecomposition(_array_dirichlet())

    assert isinstance(decomposition.epistemic, np.ndarray)


def test_array_decomposition_uniform_dirichlet_has_max_vacuity() -> None:
    """Sensoy: total uncertainty / vacuity is 1 when there is no evidence beyond the uniform prior."""
    uniform = ArrayDirichletDistribution(alphas=np.array([1.0, 1.0, 1.0, 1.0], dtype=float))

    decomposition = EvidentialClassificationDecomposition(uniform)

    np.testing.assert_allclose(decomposition.epistemic, 1.0, rtol=1e-12, atol=1e-12)
