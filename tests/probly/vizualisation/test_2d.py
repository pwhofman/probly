"""Tests for visualization for two classes."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

from probly.visualization.plot_2d import IntervalVisualizer


def test_probs_to_coords_2d_edge_values() -> None:
    """probs_to_coords_2d maps [0,1] to x=1 and y=0."""
    viz = IntervalVisualizer()
    probs = np.array([0.0, 1.0])

    x, y = viz.probs_to_coords_2d(probs)

    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)


def test_interval_plot_returns_axes_and_sets_title_and_labels() -> None:
    """interval_plot returns Axes, sets title, and uses the first two labels."""
    viz = IntervalVisualizer()
    probs = np.array([[0.4, 0.6], [0.2, 0.8]])
    labels = ["C1", "C2", "C3", "C4"]
    title = "My Interval Plot"

    ax = viz.interval_plot(probs, labels=labels, title=title)

    assert ax is not None
    assert ax.get_title() == title

    texts = [t.get_text() for t in ax.texts]
    assert "C1" in texts
    assert "C2" in texts
    assert "C3" not in texts
    assert "C4" not in texts


def test_interval_plot_adds_legend_entry() -> None:
    """interval_plot adds a legend entry for the scatter points."""
    viz = IntervalVisualizer()
    probs = np.array([[0.4, 0.6]])
    ax = viz.interval_plot(probs, labels=["C1", "C2"])

    legend = ax.get_legend()
    assert legend is not None

    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Probabilities" in legend_texts


def test_interval_plot_raises_if_labels_too_short() -> None:
    """interval_plot fails if fewer than two labels are provided (current behavior)."""
    viz = IntervalVisualizer()
    probs = np.array([[0.4, 0.6]])

    with pytest.raises(IndexError):
        viz.interval_plot(probs, labels=["C1"])
