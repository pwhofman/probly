"""Integration tests to check front to end functionality."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

from pathlib import Path
from typing import cast

import matplotlib.axes as mplaxes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from probly.visualization.credal.credal_visualization import create_credal_plot


def _assert_plot_created(ax: mplaxes.Axes | None) -> None:
    """Assert that a plot was created (Axes returned or at least a Figure exists)."""
    if ax is not None:
        assert isinstance(ax, mplaxes.Axes)
        ax.figure.clf()
        return

    # Fallback: if no Axes was returned, check that a matplotlib Figure exists.
    assert len(plt.get_fignums()) > 0, "Plot returned None and no figure was created."
    plt.close("all")


def test_integration_2_classes_interval_plot() -> None:
    """Integration test.

    2 classes should trigger the 2D interval plot.
    """
    data = np.array([[0.5, 0.5], [0.2, 0.8]])
    ax = cast(mplaxes.Axes | None, create_credal_plot(data))
    _assert_plot_created(ax)


def test_integration_3_classes_ternary_plot() -> None:
    """Integration test.

    3 classes should trigger the ternary plot.
    """
    data = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    ax = cast(mplaxes.Axes | None, create_credal_plot(data))
    _assert_plot_created(ax)


def test_integration_5_classes_spider_plot() -> None:
    """Integration test.

    More than 3 classes should trigger the multi/spider plot.
    """
    data = np.array(
        [
            [0.2, 0.3, 0.1, 0.15, 0.25],
            [0.05, 0.4, 0.2, 0.1, 0.25],
        ],
    )
    ax = cast(mplaxes.Axes | None, create_credal_plot(data))
    _assert_plot_created(ax)


def test_integration_plot_can_be_saved(tmp_path: Path) -> None:
    """Integration test.

    The created plot should be saveable via savefig.
    """
    data = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    ax = cast(mplaxes.Axes | None, create_credal_plot(data))

    fig = cast(Figure, ax.figure if ax is not None else plt.gcf())

    out = tmp_path / "credal_plot.png"
    fig.savefig(out)

    assert out.exists()
    assert out.stat().st_size > 0

    plt.close("all")
