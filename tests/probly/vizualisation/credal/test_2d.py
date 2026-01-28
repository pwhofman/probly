"""Tests for visualization for two classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from probly.visualization.credal.plot_2d import IntervalVisualizer


def _legend_labels(ax: Axes) -> list[str]:
    """Return legend labels (empty list if no legend)."""
    leg = ax.get_legend()
    if leg is None:
        return []
    return [t.get_text() for t in leg.get_texts()]


@pytest.fixture
def viz() -> IntervalVisualizer:
    return IntervalVisualizer()


@pytest.fixture
def labels() -> list[str]:
    return ["A", "B"]


@pytest.fixture
def example_probs() -> np.ndarray:
    """Example probabilities for testing flags."""
    return np.array(
        [
            [0.4, 0.6],
            [0.2, 0.8],
            [0.7, 0.3],
        ],
        dtype=float,
    )


def test_probs_to_coords_2d_example_point(viz: IntervalVisualizer) -> None:
    """One example point gets mapped correctly."""
    probs = np.array([0.3, 0.7])
    x, y = viz.probs_to_coords_2d(probs)

    assert x == pytest.approx(0.7)
    assert y == pytest.approx(0.0)


def test_probs_to_coords_2d_maps_vertices_to_endpoints(viz: IntervalVisualizer) -> None:
    """Extreme probabilities map to interval endpoints."""
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.0, 1.0])

    x1, y1 = viz.probs_to_coords_2d(p1)
    x2, y2 = viz.probs_to_coords_2d(p2)

    assert x1 == pytest.approx(0.0)
    assert y1 == pytest.approx(0.0)

    assert x2 == pytest.approx(1.0)
    assert y2 == pytest.approx(0.0)


def test_interval_plot_returns_axes_and_sets_title(
    viz: IntervalVisualizer, example_probs: np.ndarray, labels: list[str]
) -> None:
    """interval_plot returns Axes and sets the provided title."""
    fig, ax = plt.subplots()
    title = "My Interval Plot"

    ax_out = viz.interval_plot(
        example_probs,
        labels=labels,
        title=title,
        mle_flag=False,
        credal_flag=False,
        ax=ax,
    )

    assert ax_out is ax
    assert ax.get_title() == title

    plt.close(fig)


def test_interval_plot_uses_custom_labels_for_endpoints(viz: IntervalVisualizer, example_probs: np.ndarray) -> None:
    """Custom labels appear in the plot texts."""
    labels = ["Left", "Right"]

    ax = viz.interval_plot(
        example_probs,
        labels=labels,
        title="Title",
        mle_flag=False,
        credal_flag=False,
    )

    texts = [t.get_text() for t in ax.texts]
    assert "Left" in texts
    assert "Right" in texts


def test_interval_plot_raises_if_label_count_too_small(viz: IntervalVisualizer) -> None:
    """Too few labels triggers an IndexError (current behavior)."""
    probs = np.array([[0.4, 0.6]])

    with pytest.raises(IndexError):
        viz.interval_plot(
            probs,
            labels=["OnlyOne"],
            title="Title",
            mle_flag=False,
            credal_flag=False,
        )


def test_legend_probabilities_only_when_all_flags_false(
    viz: IntervalVisualizer, example_probs: np.ndarray, labels: list[str]
) -> None:
    """Legend only contains probabilities when no flags are set."""
    fig, ax = plt.subplots()

    viz.interval_plot(
        example_probs,
        labels=labels,
        title="Title",
        mle_flag=False,
        credal_flag=False,
        ax=ax,
    )

    labs = _legend_labels(ax)
    assert "Probabilities" in labs
    assert "MLE" not in labs
    assert "Credal-band" not in labs

    plt.close(fig)


def test_legend_only_mle(viz: IntervalVisualizer, example_probs: np.ndarray, labels: list[str]) -> None:
    """Legend contains probabilities and MLE only."""
    fig, ax = plt.subplots()

    viz.interval_plot(
        example_probs,
        labels=labels,
        title="Title",
        mle_flag=True,
        credal_flag=False,
        ax=ax,
    )

    labs = _legend_labels(ax)
    assert "Probabilities" in labs
    assert "MLE" in labs
    assert "Credal-band" not in labs

    plt.close(fig)


def test_legend_only_credal(viz: IntervalVisualizer, example_probs: np.ndarray, labels: list[str]) -> None:
    """Legend contains probabilities and credal band only."""
    fig, ax = plt.subplots()

    viz.interval_plot(
        example_probs,
        labels=labels,
        title="Title",
        mle_flag=False,
        credal_flag=True,
        ax=ax,
    )

    labs = _legend_labels(ax)
    assert "Probabilities" in labs
    assert "MLE" not in labs
    assert "Credal-band" in labs

    plt.close(fig)


def test_legend_includes_all_when_all_flags_true(
    viz: IntervalVisualizer, example_probs: np.ndarray, labels: list[str]
) -> None:
    """Legend contains all entries when all flags are enabled."""
    fig, ax = plt.subplots()

    viz.interval_plot(
        example_probs,
        labels=labels,
        title="Title",
        mle_flag=True,
        credal_flag=True,
        ax=ax,
    )

    labs = _legend_labels(ax)
    assert "Probabilities" in labs
    assert "MLE" in labs
    assert "Credal-band" in labs

    plt.close(fig)
