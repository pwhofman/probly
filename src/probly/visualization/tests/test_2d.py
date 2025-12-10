"""Test for visualization for two classes."""

from __future__ import annotations

import numpy as np
import pytest

from probly.visualization.plot_2d import IntervalVisualizer


def test_edge_values() -> None:
    """Testing if edge values are working correctly."""
    viz = IntervalVisualizer()
    probs = np.array([0.0, 1.0])

    x, y = viz.probs_to_coords_2d(probs)

    assert x == pytest.approx(1.0)  # noqa: S101
    assert y == pytest.approx(0.0)  # noqa: S101


def test_raise_with_nonmatching_classes() -> None:
    """Testing if more clases are raising error."""
    viz = IntervalVisualizer()
    probs = np.array([[0.4, 0.6]])
    with pytest.raises(ValueError, match=r"Number of labels .* must match number of classes"):
        viz.interval_plot(probs, labels=["C1", "C2", "C3", "C4"])
