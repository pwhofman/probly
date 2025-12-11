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


def test_interval_plot_uses_two_lables_accepted() -> None:
    """Interval plot should ignore extra labels and still work."""
    viz = IntervalVisualizer()
    probs = np.array([[0.4, 0.6]])

    ax = viz.interval_plot(probs, labels=["C1", "C2", "C3", "C4"])
    assert ax is not None  # noqa: S101

    texts = [t.get_text() for t in ax.texts]

    assert "C1" in texts  # noqa: S101
    assert "C2" in texts  # noqa: S101
