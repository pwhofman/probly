"""Tests for visualization for three classes."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pytest

from probly.visualization.plot_3d import TernaryVisualizer


def test_probs_to_coords_3d_maps_vertices_to_triangle_corners() -> None:
    """Vertices map to the three triangle corners in 2D."""
    viz = TernaryVisualizer()

    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 1.0])

    x1, y1 = viz.probs_to_coords_3d(p1)
    x2, y2 = viz.probs_to_coords_3d(p2)
    x3, y3 = viz.probs_to_coords_3d(p3)

    assert x1 == pytest.approx(0.0)
    assert y1 == pytest.approx(0.0)

    assert x2 == pytest.approx(1.0)
    assert y2 == pytest.approx(0.0)

    assert x3 == pytest.approx(0.5)
    assert y3 == pytest.approx(np.sqrt(3) / 2)


def test_ternary_plot_uses_custom_labels_for_vertices() -> None:
    """Custom labels appear in the plot texts."""
    viz = TernaryVisualizer()
    probs = np.array([[0.2, 0.3, 0.5]])
    labels = ["A", "B", "C"]

    ax = viz.ternary_plot(probs, labels=labels, plot_hull=False)

    texts = [t.get_text() for t in ax.texts]
    assert "A" in texts
    assert "B" in texts
    assert "C" in texts


def test_ternary_plot_raises_if_label_count_too_small() -> None:
    """Too few labels triggers an IndexError (current behavior)."""
    viz = TernaryVisualizer()
    probs = np.array([[0.2, 0.3, 0.5]])

    with pytest.raises(IndexError):
        viz.ternary_plot(probs, labels=["C1", "C2"], plot_hull=False)


def test_ternary_plot_sets_title() -> None:
    """ternary_plot sets the provided title on the axes."""
    viz = TernaryVisualizer()
    probs = np.array([[0.2, 0.3, 0.5]])
    labels = ["A", "B", "C"]
    title = "My Ternary Plot"

    ax = viz.ternary_plot(probs, labels=labels, title=title, plot_hull=False)

    assert ax.get_title() == title


@pytest.mark.parametrize(
    "probs",
    [
        np.array([[0.7, 0.2, 0.1]]),  # single point
        np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),  # two points
        np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.8, 0.1, 0.1],
                [0.3, 0.1, 0.6],
                [0.33, 0.33, 0.34],
            ],
        ),  # normal case (>=3 points)
    ],
)
def test_plot_convex_hull_runs_without_error(probs: np.ndarray) -> None:
    """plot_convex_hull runs without crashing for degenerate and normal inputs."""
    viz = TernaryVisualizer()
    ax = viz.plot_convex_hull(probs)

    assert ax is not None
