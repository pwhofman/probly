"""Tests for ternary credal plotting in ``probly.plot.credal._ternary``."""

from __future__ import annotations

import numpy as np
import pytest

from probly.plot import plot_credal_set
from probly.representation.credal_set.numpy import (
    NumpyConvexCredalSet,
    NumpyDiscreteCredalSet,
    NumpyDistanceBasedCredalSet,
    NumpyProbabilityIntervalsCredalSet,
    NumpySingletonCredalSet,
)
from probly.representation.distribution.numpy_categorical import (
    NumpyProbabilityCategoricalDistribution,
)


@pytest.mark.usefixtures("_close_figures")
class TestTernaryPlot:
    """Ternary (3-class) credal plotting."""

    def test_singleton_ternary(self) -> None:
        data = NumpySingletonCredalSet(
            array=NumpyProbabilityCategoricalDistribution(np.array([[0.5, 0.3, 0.2]])),
        )
        ax = plot_credal_set(data, title="Tern Singleton")
        # mpltern axes have name 'ternary'
        assert "ternary" in ax.name

    def test_intervals_ternary(self) -> None:
        data = NumpyProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.1, 0.1]]),
            upper_bounds=np.array([[0.7, 0.7, 0.7]]),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_intervals_ternary_zero_width_renders_point(self) -> None:
        # All bounds equal -> single feasible vertex
        data = NumpyProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.4, 0.3, 0.3]]),
            upper_bounds=np.array([[0.4, 0.3, 0.3]]),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_distance_based_ternary(self) -> None:
        data = NumpyDistanceBasedCredalSet(
            nominal=np.array([[0.4, 0.3, 0.3]]),
            radius=np.array([0.1]),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_convex_ternary(self) -> None:
        data = NumpyConvexCredalSet(
            array=np.array([[[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]]),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_convex_ternary_two_points(self) -> None:
        data = NumpyConvexCredalSet(
            array=np.array([[[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]]]),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_convex_ternary_single_point(self) -> None:
        data = NumpyConvexCredalSet(
            array=np.array([[[0.6, 0.2, 0.2]]]),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_discrete_ternary(self) -> None:
        data = NumpyDiscreteCredalSet(
            array=np.array([[[0.5, 0.3, 0.2], [0.3, 0.5, 0.2]]]),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_labels_mismatch_raises(self) -> None:
        data = NumpySingletonCredalSet(array=np.array([[0.4, 0.3, 0.3]]))
        with pytest.raises(ValueError, match="Expected 3 labels"):
            plot_credal_set(data, labels=["A", "B"])

    def test_ground_truth_ternary(self) -> None:
        data = NumpySingletonCredalSet(array=np.array([[0.5, 0.3, 0.2]]))
        gt = NumpySingletonCredalSet(array=np.array([[0.0, 0.0, 1.0]]))
        ax = plot_credal_set(data, ground_truth=gt)
        assert "ternary" in ax.name

    def test_gridlines_off(self) -> None:
        data = NumpySingletonCredalSet(array=np.array([[0.5, 0.3, 0.2]]))
        ax = plot_credal_set(data, gridlines=False)
        assert "ternary" in ax.name
