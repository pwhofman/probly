"""Tests for conformal-set uncertainty measures."""

from __future__ import annotations

import numpy as np

from probly.quantification import decompose, measure, quantify
from probly.quantification.decomposition.decomposition import ConstantTotalDecomposition
from probly.representation.conformal_set import ArrayIntervalConformalSet, ArrayOneHotConformalSet


def test_measure_uses_set_size_for_array_one_hot_conformal_set() -> None:
    conformal_set = ArrayOneHotConformalSet(
        np.array(
            [
                [True, False, True],
                [False, True, False],
            ],
        )
    )

    uncertainty = measure(conformal_set)

    np.testing.assert_array_equal(uncertainty, conformal_set.set_size)


def test_measure_uses_set_size_for_array_interval_conformal_set() -> None:
    lower = np.array([0.0, 2.0, -1.0])
    upper = np.array([1.5, 5.0, 2.0])
    conformal_set = ArrayIntervalConformalSet.from_array_samples(lower, upper)

    uncertainty = measure(conformal_set)

    np.testing.assert_allclose(uncertainty, conformal_set.set_size, rtol=1e-12, atol=1e-12)


def test_decompose_wraps_array_conformal_set_size_as_total_uncertainty() -> None:
    conformal_set = ArrayOneHotConformalSet(
        np.array(
            [
                [True, False, True],
                [True, True, True],
            ],
        )
    )

    decomposition = decompose(conformal_set)

    assert isinstance(decomposition, ConstantTotalDecomposition)
    np.testing.assert_array_equal(decomposition.total, conformal_set.set_size)


def test_quantify_uses_decompose_wrapper_for_array_conformal_set() -> None:
    lower = np.array([[0.0, 1.0], [2.0, 3.0]])
    upper = np.array([[0.5, 2.5], [5.0, 4.0]])
    conformal_set = ArrayIntervalConformalSet.from_array_samples(lower, upper)

    quantification = quantify(conformal_set)

    assert isinstance(quantification, ConstantTotalDecomposition)
    np.testing.assert_allclose(quantification.total, conformal_set.set_size, rtol=1e-12, atol=1e-12)
