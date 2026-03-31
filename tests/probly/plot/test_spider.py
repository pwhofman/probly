"""Tests for spider (radar) credal set plotting."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from probly.plot import plot_credal_set
from probly.plot.credal._spider import _ray_segment_r, _segment_intersection
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)

mpl.use("Agg")

NUM_CLASSES = 5


@pytest.fixture
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.usefixtures("_close_figures")
class TestSpiderPlot:
    def test_singleton(self):
        data = ArraySingletonCredalSet(
            array=np.array([[0.3, 0.2, 0.1, 0.15, 0.25]]),
        )
        ax = plot_credal_set(data, title="Singleton Spider")
        assert f"radar_{NUM_CLASSES}" in ax.name

    def test_probability_intervals(self):
        data = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.05, 0.05, 0.05, 0.05, 0.05]]),
            upper_bounds=np.array([[0.4, 0.3, 0.3, 0.3, 0.5]]),
        )
        ax = plot_credal_set(data, title="Intervals Spider")
        assert f"radar_{NUM_CLASSES}" in ax.name

    def test_distance_based(self):
        data = ArrayDistanceBasedCredalSet(
            nominal=np.array([[0.3, 0.2, 0.2, 0.15, 0.15]]),
            radius=0.05,
        )
        ax = plot_credal_set(data, title="Distance-Based Spider")
        assert f"radar_{NUM_CLASSES}" in ax.name

    def test_convex(self):
        data = ArrayConvexCredalSet(
            array=np.array(
                [
                    [
                        [0.5, 0.2, 0.1, 0.1, 0.1],
                        [0.1, 0.5, 0.2, 0.1, 0.1],
                        [0.1, 0.1, 0.5, 0.2, 0.1],
                    ],
                ]
            ),
        )
        ax = plot_credal_set(data, title="Convex Spider")
        assert f"radar_{NUM_CLASSES}" in ax.name

    def test_discrete(self):
        data = ArrayDiscreteCredalSet(
            array=np.array(
                [
                    [
                        [0.4, 0.2, 0.15, 0.15, 0.1],
                        [0.2, 0.4, 0.1, 0.15, 0.15],
                    ],
                ]
            ),
        )
        ax = plot_credal_set(data, title="Discrete Spider")
        assert f"radar_{NUM_CLASSES}" in ax.name

    def test_batched_input(self):
        data = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.array(
                [
                    [0.05, 0.05, 0.05, 0.05, 0.05],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                ]
            ),
            upper_bounds=np.array(
                [
                    [0.4, 0.3, 0.3, 0.3, 0.5],
                    [0.3, 0.25, 0.25, 0.25, 0.35],
                ]
            ),
        )
        ax = plot_credal_set(
            data,
            title="Batched",
            series_labels=["Set A", "Set B"],
        )
        assert f"radar_{NUM_CLASSES}" in ax.name

    def test_labels_mismatch_raises(self):
        data = ArraySingletonCredalSet(
            array=np.array([[0.3, 0.2, 0.1, 0.15, 0.25]]),
        )
        with pytest.raises(ValueError, match="Expected 5 labels"):
            plot_credal_set(data, labels=["A", "B", "C"])

    def test_gridlines_off(self):
        data = ArraySingletonCredalSet(
            array=np.array([[0.3, 0.2, 0.1, 0.15, 0.25]]),
        )
        ax = plot_credal_set(data, gridlines=False)
        assert f"radar_{NUM_CLASSES}" in ax.name

    def test_custom_labels(self):
        data = ArraySingletonCredalSet(
            array=np.array([[0.3, 0.2, 0.1, 0.15, 0.25]]),
        )
        labels = ["Cat", "Dog", "Bird", "Fish", "Frog"]
        ax = plot_credal_set(data, labels=labels)
        assert f"radar_{NUM_CLASSES}" in ax.name

    def test_eight_classes(self):
        nc = 8
        data = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.full((1, nc), 0.05),
            upper_bounds=np.full((1, nc), 0.3),
        )
        ax = plot_credal_set(data, title="8-class spider")
        assert f"radar_{nc}" in ax.name


@pytest.mark.usefixtures("_close_figures")
class TestGroundTruthOverlay:
    def test_ground_truth_spider(self):
        data = ArraySingletonCredalSet(
            array=np.array([[0.3, 0.2, 0.1, 0.15, 0.25]]),
        )
        gt = ArraySingletonCredalSet(array=np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]))
        ax = plot_credal_set(data, ground_truth=gt)
        assert f"radar_{NUM_CLASSES}" in ax.name

    def test_ground_truth_with_intervals(self):
        data = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.05, 0.05, 0.05, 0.05, 0.05]]),
            upper_bounds=np.array([[0.4, 0.3, 0.3, 0.3, 0.5]]),
        )
        gt = ArraySingletonCredalSet(array=np.array([[0.3, 0.2, 0.2, 0.15, 0.15]]))
        ax = plot_credal_set(data, ground_truth=gt)
        assert f"radar_{NUM_CLASSES}" in ax.name

    def test_ground_truth_binary(self):
        data = ArraySingletonCredalSet(array=np.array([[0.3, 0.7]]))
        gt = ArraySingletonCredalSet(array=np.array([[0.0, 1.0]]))
        ax = plot_credal_set(data, ground_truth=gt)
        assert ax is not None

    def test_ground_truth_ternary(self):
        data = ArraySingletonCredalSet(array=np.array([[0.5, 0.3, 0.2]]))
        gt = ArraySingletonCredalSet(array=np.array([[0.0, 1.0, 0.0]]))
        ax = plot_credal_set(data, ground_truth=gt)
        assert ax is not None


class TestSegmentIntersection:
    """Unit tests for _segment_intersection."""

    def test_crossing_segments(self):

        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 1.0])
        p3 = np.array([0.0, 1.0])
        p4 = np.array([1.0, 0.0])
        result = _segment_intersection(p1, p2, p3, p4)
        assert result is not None
        np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-10)

    def test_parallel_segments(self):

        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([0.0, 1.0])
        p4 = np.array([1.0, 1.0])
        assert _segment_intersection(p1, p2, p3, p4) is None

    def test_non_crossing_segments(self):

        # Segments that would cross if extended but don't overlap
        p1 = np.array([0.0, 0.0])
        p2 = np.array([0.4, 0.4])
        p3 = np.array([0.6, 0.0])
        p4 = np.array([1.0, 0.4])
        assert _segment_intersection(p1, p2, p3, p4) is None


class TestRaySegmentR:
    """Unit tests for _ray_segment_r."""

    def test_ray_hits_segment(self):

        # Ray along x-axis hitting a vertical segment at x=1
        p_start = np.array([1.0, -1.0])
        p_end = np.array([1.0, 1.0])
        result = _ray_segment_r(1.0, 0.0, p_start, p_end)
        assert result is not None
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_ray_parallel_to_segment(self):

        # Ray along x-axis, segment along x-axis
        p_start = np.array([1.0, 0.0])
        p_end = np.array([2.0, 0.0])
        assert _ray_segment_r(1.0, 0.0, p_start, p_end) is None

    def test_ray_misses_segment(self):

        # Ray along x-axis, segment entirely in y>0 region far away
        p_start = np.array([0.0, 2.0])
        p_end = np.array([1.0, 2.0])
        assert _ray_segment_r(1.0, 0.0, p_start, p_end) is None


@pytest.mark.usefixtures("_close_figures")
class TestEnvelopeGeometry:
    """Test that the convex spider envelope produces a closed fill region."""

    def test_envelope_with_known_crossing(self):
        # Two members on 4 spokes where member lines cross between spokes
        data = ArrayConvexCredalSet(
            array=np.array(
                [
                    [
                        [0.4, 0.1, 0.3, 0.2],
                        [0.1, 0.4, 0.2, 0.3],
                    ],
                ]
            ),
        )
        ax = plot_credal_set(data, envelope=True, title="Envelope crossing test")
        # The plot should have fill artists from the envelope
        fills = [c for c in ax.get_children() if hasattr(c, "get_facecolor")]
        assert len(fills) > 0
