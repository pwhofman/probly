"""Tests for numpy credal set uncertainty measures."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from probly.quantification import quantify
from probly.quantification.measure.credal_set import lower_entropy, upper_entropy
from probly.representation.credal_set.numpy import NumpyProbabilityIntervalsCredalSet

from ._entropy_suite import (
    DTYPES,
    IntervalLowerEntropySuite,
    assert_in_intervals,
    entropy,
    random_intervals,
    tolerance,
)


class _NumpyBackend:
    """Builds numpy credal sets from numpy arrays for the shared correctness suite."""

    @staticmethod
    def intervals(lower: np.ndarray, upper: np.ndarray) -> NumpyProbabilityIntervalsCredalSet:
        return NumpyProbabilityIntervalsCredalSet(lower_bounds=np.asarray(lower), upper_bounds=np.asarray(upper))

    @staticmethod
    def numpy(value: np.ndarray) -> np.ndarray:
        return np.asarray(value)


@pytest.fixture
def backend() -> _NumpyBackend:
    return _NumpyBackend()


def _gradient(p: np.ndarray) -> np.ndarray:
    """Entropy gradient ``-log(p) - 1``, with the log of exact zeros taken at the smallest positive float."""
    return -np.log(np.maximum(np.asarray(p, dtype=np.float64), np.finfo(np.float64).tiny)) - 1.0


# ---------------------------------------------------------------------------
# Probability intervals
# ---------------------------------------------------------------------------


class TestNumpyIntervalLowerEntropy(IntervalLowerEntropySuite):
    """Probability-interval lower entropy on numpy arrays."""


@DTYPES
@pytest.mark.parametrize("n_classes", [2, 3, 5, 10, 30])
def test_intervals_upper_entropy_is_the_maximum(dtype: type[np.floating], n_classes: int) -> None:
    """No point of the polytope improves the linear bound ``H(q) <= H(p) + g . (q - p)`` at the maximizer."""
    rng = np.random.default_rng(n_classes)
    bounds = [random_intervals(rng, kind, n_classes) for kind in ["ensemble", "dirichlet", "sparse", "box"] * 2]
    lower = np.stack([b[0] for b in bounds]).astype(dtype)
    upper = np.stack([b[1] for b in bounds]).astype(dtype)

    value, p = upper_entropy(_NumpyBackend.intervals(lower, upper), return_distribution=True)

    assert value.dtype == p.dtype == dtype
    assert_in_intervals(p, lower, upper, atol=tolerance(dtype))
    np.testing.assert_allclose(entropy(p), value, atol=tolerance(dtype))
    certificate = 1e-9 if dtype == np.float64 else 1e-4
    for row in range(len(bounds)):
        gradient = _gradient(p[row])
        result = linprog(
            -gradient, A_eq=np.ones((1, n_classes)), b_eq=[1.0], bounds=list(zip(lower[row], upper[row], strict=True))
        )
        assert -result.fun - gradient @ p[row] <= certificate, row


def test_intervals_upper_entropy_known_values() -> None:
    """A singleton gives its own entropy, the full simplex the uniform distribution."""
    point = np.array([0.2, 0.5, 0.3])
    credal_set = _NumpyBackend.intervals(np.stack([point, np.zeros(3)]), np.stack([point, np.ones(3)]))

    value, p = upper_entropy(credal_set, return_distribution=True)

    np.testing.assert_allclose(value, [entropy(point), np.log(3)], atol=1e-12)
    np.testing.assert_allclose(p, [point, np.full(3, 1 / 3)], atol=1e-12)
    np.testing.assert_allclose(upper_entropy(credal_set, base="normalize"), value / np.log(3))
    np.testing.assert_allclose(upper_entropy(credal_set, base=2.0), value / np.log(2))


def test_intervals_quantify_decomposes_into_upper_and_lower_entropy() -> None:
    """``quantify`` on numpy probability intervals gives total = upper and aleatoric = lower entropy."""
    rng = np.random.default_rng(0)
    bounds = [random_intervals(rng, "ensemble", 4) for _ in range(5)]
    credal_set = _NumpyBackend.intervals(np.stack([b[0] for b in bounds]), np.stack([b[1] for b in bounds]))

    decomposition = quantify(credal_set)

    np.testing.assert_allclose(decomposition.total, upper_entropy(credal_set))
    np.testing.assert_allclose(decomposition.aleatoric, lower_entropy(credal_set))
    np.testing.assert_allclose(decomposition.epistemic, decomposition.total - decomposition.aleatoric)
    assert np.all(decomposition.epistemic >= 0.0)
