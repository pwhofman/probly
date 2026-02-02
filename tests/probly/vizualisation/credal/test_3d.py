"""Tests for visualization for three classes."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest

if TYPE_CHECKING:
    from matplotlib.axes import Axes
from probly.visualization.credal.plot_3d import TernaryVisualizer


def _legend_labels(ax: Axes) -> list[str]:
    """Return legend labels (empty list if no legend)."""
    leg = ax.get_legend()
    if leg is None:
        return []
    return [t.get_text() for t in leg.get_texts()]


@pytest.fixture
def viz() -> TernaryVisualizer:
    return TernaryVisualizer()


@pytest.fixture
def labels() -> list[str]:
    return ["A", "B", "C"]


@pytest.fixture
def example_probs() -> np.ndarray:
    """Example probs for testing flags."""
    return np.array(
        [
            [0.20, 0.30, 0.50],
            [0.25, 0.35, 0.40],
            [0.30, 0.25, 0.45],
            [0.35, 0.30, 0.35],
        ],
        dtype=float,
    )


def test_legend_probabilities_only_when_all_flags_false(
    viz: TernaryVisualizer, example_probs: np.ndarray, labels: list[str]
) -> None:
    """Tests if legend stays correct when no other label given."""
    fig, ax = plt.subplots()

    ax_out = viz.ternary_plot(
        example_probs,
        labels=labels,
        title="Title",
        credal_flag=False,
        mle_flag=False,
        minmax_flag=False,
        ax=ax,
    )
    assert ax_out is ax

    labs = _legend_labels(ax)
    assert "Probabilities" in labs
    assert "MLE" not in labs
    assert "Credal set" not in labs
    assert "Min envelope" not in labs
    assert "Max envelope" not in labs

    plt.close(fig)


def test_legend_uses_custom_probs_label(viz: TernaryVisualizer, example_probs: np.ndarray, labels: list[str]) -> None:
    """Tests if custom label is used."""
    fig, ax = plt.subplots()

    viz.ternary_plot(
        example_probs,
        labels=labels,
        title="Title",
        credal_flag=False,
        mle_flag=False,
        minmax_flag=False,
        ax=ax,
        label="TestLabel",
    )

    labs = _legend_labels(ax)
    assert "TestLabel" in labs
    assert "Probabilities" not in labs

    plt.close(fig)


def test_legend_includes_all_when_all_flags_true(
    viz: TernaryVisualizer, example_probs: np.ndarray, labels: list[str]
) -> None:
    """Tests if all labels are working and min/max only shows once."""
    fig, ax = plt.subplots()

    viz.ternary_plot(
        example_probs,
        labels=labels,
        title="Title",
        credal_flag=True,
        mle_flag=True,
        minmax_flag=True,
        ax=ax,
    )

    labs = _legend_labels(ax)
    assert "Probabilities" in labs
    assert "MLE" in labs
    assert "Credal set" in labs
    assert labs.count("Min envelope") == 1
    assert labs.count("Max envelope") == 1

    plt.close(fig)


def test_legend_only_mle(viz: TernaryVisualizer, example_probs: np.ndarray, labels: list[str]) -> None:
    """Tests if all labels are working and min/max only shows once."""
    fig, ax = plt.subplots()

    viz.ternary_plot(
        example_probs,
        labels=labels,
        title="Title",
        credal_flag=False,
        mle_flag=True,
        minmax_flag=False,
        ax=ax,
    )

    labs = _legend_labels(ax)
    assert "Probabilities" in labs
    assert "MLE" in labs
    assert "Credal set" not in labs
    assert "Min envelope" not in labs
    assert "Max envelope" not in labs

    plt.close(fig)


def test_legend_only_credalhull(viz: TernaryVisualizer, example_probs: np.ndarray, labels: list[str]) -> None:
    """Tests if labels are working if credalhull only true."""
    fig, ax = plt.subplots()

    viz.ternary_plot(
        example_probs,
        labels=labels,
        title="Title",
        credal_flag=True,
        mle_flag=False,
        minmax_flag=False,
        ax=ax,
    )

    labs = _legend_labels(ax)
    assert "Probabilities" in labs
    assert "MLE" not in labs
    assert "Credal set" in labs
    assert "Min envelope" not in labs
    assert "Max envelope" not in labs

    plt.close(fig)


def test_legend_only_minmax(viz: TernaryVisualizer, example_probs: np.ndarray, labels: list[str]) -> None:
    """Tests if labels are working for min max."""
    fig, ax = plt.subplots()

    viz.ternary_plot(
        example_probs,
        labels=labels,
        title="Title",
        credal_flag=False,
        mle_flag=False,
        minmax_flag=True,
        ax=ax,
    )

    labs = _legend_labels(ax)
    assert "Probabilities" in labs
    assert "MLE" not in labs
    assert "Credal set" not in labs
    assert labs.count("Min envelope") == 1
    assert labs.count("Max envelope") == 1

    plt.close(fig)


def test_plot_convex_hull_collinear_points_falls_back_to_line() -> None:
    """Colinear points should not crash and should draw a line."""
    viz = TernaryVisualizer()

    ts = np.linspace(0.1, 0.9, 10)
    probs = np.array([[0.0, float(t), float(1.0 - t)] for t in ts])

    ax = viz.plot_convex_hull(probs)

    assert len(ax.lines) >= 1
    assert len(ax.patches) == 0


def test_probs_to_coords_3d_example_point() -> None:
    """One example point gets mapped correctly."""
    viz = TernaryVisualizer()
    probs = np.array([0.2, 0.3, 0.5])
    x, y = viz.probs_to_coords_3d(probs)

    assert x == pytest.approx(0.55)
    assert y == pytest.approx(np.sqrt(3) / 4)


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

    ax = viz.ternary_plot(
        probs,
        labels=labels,
        title="Title",
        credal_flag=False,
        mle_flag=False,
        minmax_flag=False,
    )

    texts = [t.get_text() for t in ax.texts]
    assert "A" in texts
    assert "B" in texts
    assert "C" in texts


def test_ternary_plot_raises_if_label_count_too_small() -> None:
    """Too few labels triggers an IndexError (current behavior)."""
    viz = TernaryVisualizer()
    probs = np.array([[0.2, 0.3, 0.5]])

    with pytest.raises(IndexError):
        viz.ternary_plot(
            probs,
            labels=["C1", "C2"],
            title="Title",
            credal_flag=False,
            mle_flag=False,
            minmax_flag=False,
        )


def test_ternary_plot_sets_title() -> None:
    """ternary_plot sets the provided title on the axes."""
    viz = TernaryVisualizer()
    probs = np.array([[0.2, 0.3, 0.5]])
    labels = ["A", "B", "C"]
    title = "My Ternary Plot"

    ax = viz.ternary_plot(
        probs,
        labels=labels,
        title=title,
        credal_flag=False,
        mle_flag=False,
        minmax_flag=False,
    )

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
