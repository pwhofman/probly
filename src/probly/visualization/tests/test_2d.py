"""Test for visualization for two classes."""

from __future__ import annotations

import numpy as np
import pytest

from probly.visualization.plot_2d import *  # noqa: F403


def test_edge_values() -> None:

    viz = IntervalVisualizer()
    probs = np.array([0.0, 1.0])

    x, y = viz.probs_to_coords_2d(probs)

    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)


def test_raise_with_nonmatching_classes() -> None:
    viz = IntervalVisualizer()
    probs = np.array([[0.4, 0.6]])
    with pytest.raises(ValueError):
        viz.interval_plot(probs, labels=["C1","C2","C3","C4"])

