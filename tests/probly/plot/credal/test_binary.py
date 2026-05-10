"""Tests for binary credal plotting in ``probly.plot.credal._binary``."""

from __future__ import annotations

import numpy as np
import pytest

from probly.plot import plot_credal_set
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)


@pytest.mark.usefixtures("_close_figures")
class TestBinaryPlot:
    """Binary (2-class) credal plotting."""

    def test_singleton_binary(self) -> None:
        data = ArraySingletonCredalSet(array=np.array([[0.3, 0.7]]))
        ax = plot_credal_set(data, title="Singleton")
        # The binary axis goes [0, 1]
        assert ax.get_xlim() == (0.0, 1.0)
        assert ax.get_title() == "Singleton"

    def test_intervals_binary_default_labels(self) -> None:
        data = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.3]]),
            upper_bounds=np.array([[0.6, 0.8]]),
        )
        ax = plot_credal_set(data)
        assert ax.get_xlabel() == "Probability of Class 1"

    def test_intervals_binary_zero_width_renders_point(self) -> None:
        # When lo == hi, the binary plot draws a scatter dot.
        data = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.5, 0.5]]),
            upper_bounds=np.array([[0.5, 0.5]]),
        )
        ax = plot_credal_set(data)
        assert ax is not None

    def test_distance_based_binary(self) -> None:
        data = ArrayDistanceBasedCredalSet(
            nominal=np.array([[0.4, 0.6]]),
            radius=np.array([0.1]),
        )
        ax = plot_credal_set(data)
        assert ax is not None

    def test_convex_binary(self) -> None:
        data = ArrayConvexCredalSet(
            array=np.array([[[0.3, 0.7], [0.5, 0.5], [0.4, 0.6]]]),
        )
        ax = plot_credal_set(data)
        assert ax is not None

    def test_discrete_binary(self) -> None:
        data = ArrayDiscreteCredalSet(
            array=np.array([[[0.3, 0.7], [0.4, 0.6]]]),
        )
        ax = plot_credal_set(data)
        assert ax is not None

    def test_labels_mismatch_raises(self) -> None:
        data = ArraySingletonCredalSet(array=np.array([[0.3, 0.7]]))
        with pytest.raises(ValueError, match="Expected 2 labels"):
            plot_credal_set(data, labels=["only-one"])

    def test_series_labels_show_legend(self) -> None:
        data = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.3], [0.0, 0.4]]),
            upper_bounds=np.array([[0.6, 0.9], [0.5, 1.0]]),
        )
        ax = plot_credal_set(data, series_labels=["A", "B"])
        legend = ax.get_legend()
        assert legend is not None

    def test_ground_truth_binary_overlay(self) -> None:
        data = ArraySingletonCredalSet(array=np.array([[0.4, 0.6]]))
        gt = ArraySingletonCredalSet(array=np.array([[0.0, 1.0]]))
        ax = plot_credal_set(data, ground_truth=gt)
        # The legend should be present (overlay forces it on)
        assert ax.get_legend() is not None
