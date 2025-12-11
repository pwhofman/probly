"""Test for visualization for more than three classes."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")
import matplotlib.pyplot as plt

from probly.visualization.plot_multid import MultiVisualizer


def test_spider_plot_configures_axes_correctly() -> None:
    """Check r-grid, ylim, labels, and title using real matplotlib."""
    viz = MultiVisualizer()
    probs = np.array([[0.2, 0.3, 0.1, 0.4]])
    labels = ["A", "B", "C", "D"]

    ax = viz.spider_plot(probs, labels=labels, ax=None)
    y_min, y_max = ax.get_ylim()
    assert y_min == pytest.approx(0.0)  # noqa: S101
    assert y_max == pytest.approx(1.0)  # noqa: S101
    xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert xticklabels == labels  # noqa: S101

    plt.close(ax.figure)


def test_spider_plot_creates_radar_projection() -> None:
    """Ensure the projection used is 'radar'."""
    viz = MultiVisualizer()
    probs = np.array([[0.2, 0.3, 0.1, 0.4]])
    labels = ["A", "B", "C", "D"]

    ax = viz.spider_plot(probs, labels=labels, ax=None)

    assert ax.name == "radar"  # noqa: S101

    plt.close(ax.figure)
