"""NumPy credal-set tests for ``coverage`` and ``efficiency``.

Only ``ConvexCredalSet`` and ``ProbabilityIntervalsCredalSet`` are
registered. ``Singleton``, ``Discrete``, ``DistanceBased`` and
``DirichletLevelSet`` raise ``NotImplementedError`` and are intentionally
out of scope.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.metrics import coverage, efficiency
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)
from probly.representation.distribution.array_categorical import ArrayProbabilityCategoricalDistribution

from ._credal_suite import CredalSuite


def _categorical(probs: np.ndarray) -> ArrayProbabilityCategoricalDistribution:
    return ArrayProbabilityCategoricalDistribution(np.asarray(probs, dtype=float))


@pytest.fixture
def make_convex():
    return lambda probs: ArrayConvexCredalSet(array=_categorical(probs))


@pytest.fixture
def make_intervals():
    return lambda lower, upper: ArrayProbabilityIntervalsCredalSet(
        lower_bounds=np.asarray(lower, dtype=float),
        upper_bounds=np.asarray(upper, dtype=float),
    )


@pytest.fixture
def make_distribution():
    return _categorical


class TestNumpy(CredalSuite):
    """NumPy implementation of the shared credal suite."""


class TestUnregisteredCredalTypesRaise:
    """Singleton / Discrete / DistanceBased credal sets are intentionally not registered."""

    def test_singleton_raises(self) -> None:
        cs = ArraySingletonCredalSet(array=_categorical(np.array([[0.5, 0.5]])))
        with pytest.raises(NotImplementedError, match="coverage is not implemented"):
            coverage(cs, _categorical(np.array([[1.0, 0.0]])))
        with pytest.raises(NotImplementedError, match="efficiency is not implemented"):
            efficiency(cs)

    def test_discrete_raises(self) -> None:
        probs = np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]])
        cs = ArrayDiscreteCredalSet(array=_categorical(probs))
        with pytest.raises(NotImplementedError, match="coverage is not implemented"):
            coverage(cs, _categorical(np.array([[0.5, 0.4, 0.1]])))
        with pytest.raises(NotImplementedError, match="efficiency is not implemented"):
            efficiency(cs)

    def test_distance_based_raises(self) -> None:
        cs = ArrayDistanceBasedCredalSet(
            nominal=_categorical(np.array([[0.5, 0.3, 0.2]])),
            radius=np.array([0.1]),
        )
        with pytest.raises(NotImplementedError, match="coverage is not implemented"):
            coverage(cs, _categorical(np.array([[0.5, 0.3, 0.2]])))
        with pytest.raises(NotImplementedError, match="efficiency is not implemented"):
            efficiency(cs)


def test_unregistered_object_raises() -> None:
    """Falling through to the base flexdispatch raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="coverage is not implemented"):
        coverage(object(), 0)
    with pytest.raises(NotImplementedError, match="efficiency is not implemented"):
        efficiency(object())
