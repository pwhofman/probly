"""Tests for the maximin categorical decider (array backend)."""

from __future__ import annotations

import numpy as np
import pytest

from probly.decider import categorical_from_maximin
from probly.representation.credal_set.array import ArrayConvexCredalSet
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)


def test_maximin_picks_argmax_of_lower_probability_for_array_convex_credal_set() -> None:
    vertices = ArrayProbabilityCategoricalDistribution(np.array([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1]]))
    credal_set = ArrayConvexCredalSet(array=vertices)

    decision = categorical_from_maximin(credal_set)

    assert isinstance(decision, ArrayCategoricalDistribution)
    np.testing.assert_allclose(decision.probabilities, np.array([1.0, 0.0, 0.0]))


def test_maximin_handles_batched_array_convex_credal_set() -> None:
    vertices = ArrayProbabilityCategoricalDistribution(
        np.array(
            [
                [[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]],
                [[0.1, 0.7, 0.2], [0.2, 0.3, 0.5]],
            ]
        )
    )
    credal_set = ArrayConvexCredalSet(array=vertices)

    decision = categorical_from_maximin(credal_set)

    assert isinstance(decision, ArrayCategoricalDistribution)
    np.testing.assert_allclose(
        decision.probabilities,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )


def test_maximin_breaks_ties_by_picking_first_index_for_array_convex_credal_set() -> None:
    vertices = ArrayProbabilityCategoricalDistribution(np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]))
    credal_set = ArrayConvexCredalSet(array=vertices)

    decision = categorical_from_maximin(credal_set)

    np.testing.assert_allclose(decision.probabilities, np.array([1.0, 0.0, 0.0]))


def test_maximin_returns_one_hot_distribution_for_array_convex_credal_set() -> None:
    vertices = ArrayProbabilityCategoricalDistribution(
        np.array(
            [
                [[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]],
                [[0.1, 0.7, 0.2], [0.2, 0.3, 0.5]],
            ]
        )
    )
    credal_set = ArrayConvexCredalSet(array=vertices)

    decision = categorical_from_maximin(credal_set)

    np.testing.assert_allclose(decision.probabilities.sum(axis=-1), np.array([1.0, 1.0]))
    assert float(decision.probabilities.max()) == pytest.approx(1.0)
