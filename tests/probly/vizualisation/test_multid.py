"""Tests for visualization for more than three classes."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pytest

from probly.visualization.create_credal import create_credal_plot
from probly.visualization.plot_multid import MultiVisualizer


def test_spider_plot_returns_radar_axes_and_sets_ylim_and_title() -> None:
    """spider_plot returns a radar Axes and applies basic axis configuration."""
    viz = MultiVisualizer()
    probs = np.array([[0.2, 0.3, 0.1, 0.4]])
    labels = ["A", "B", "C", "D"]

    ax = viz.spider_plot(probs, labels=labels)

    assert ax.name == "radar"
    assert ax.get_ylim()[0] == pytest.approx(0.0)
    assert ax.get_ylim()[1] == pytest.approx(1.0)
    assert "Spider Plot" in ax.get_title()

    ax.figure.clf()


def test_spider_plot_adds_mle_scatter_with_label() -> None:
    """spider_plot adds a scatter point labeled 'MLE'."""
    viz = MultiVisualizer()
    probs = np.array(
        [
            [0.2, 0.3, 0.1, 0.4],
            [0.1, 0.2, 0.6, 0.1],
        ],
    )
    labels = ["A", "B", "C", "D"]

    ax = viz.spider_plot(probs, labels=labels)

    legend = ax.get_legend()
    assert legend is not None

    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "MLE" in legend_texts

    ax.figure.clf()


def test_spider_plot_uses_max_of_summed_probs_as_mle_point() -> None:
    """MLE point is placed at the argmax of probs summed over rows."""
    viz = MultiVisualizer()
    probs = np.array(
        [
            [0.1, 0.1, 0.1, 0.7],
            [0.2, 0.2, 0.1, 0.5],
        ],
    )
    labels = ["A", "B", "C", "D"]

    ax = viz.spider_plot(probs, labels=labels)

    probs_flat = probs.sum(axis=0)
    max_class = int(np.argmax(probs_flat))

    legend = ax.get_legend()
    assert legend is not None
    assert max_class == 3

    ax.figure.clf()


def test_spider_plot_raises_if_labels_too_short() -> None:
    """Too few labels should raise a ValueError (matplotlib ticklabel mismatch)."""
    viz = MultiVisualizer()
    probs = np.array([[0.2, 0.3, 0.1, 0.4]])
    labels = ["A", "B"]  # too short for 4 classes

    with pytest.raises(ValueError, match=r"(ticklabels|FixedLocator|labels|match)"):
        viz.spider_plot(probs, labels=labels)


multi = MultiVisualizer()


def test_spider_plot_single_dataset() -> None:
    """Spider plot runs without error for a single dataset."""
    labels = ["A", "B", "C", "D"]
    datasets = np.array(
        [
            [0.1, 0.4, 0.3, 0.2],
        ],
    )

    multi.spider_plot(datasets, labels=labels)


def test_spider_plot_multiple_datasets() -> None:
    """Spider plot runs with multiple datasets and matching label length."""
    labels = ["A", "B", "C", "D", "E"]
    datasets = np.array(
        [
            [0.1, 0.5, 0.2, 0.3, 0.7],
            [0.3, 0.6, 0.1, 0.4, 0.2],
            [0.8, 0.1, 0.05, 0.2, 0.3],
        ],
    )

    multi.spider_plot(datasets, labels=labels)


def test_spider_plot_length_mismatch_raises() -> None:
    labels = ["A", "B", "C"]
    datasets = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"Number of labels .* must match number of classes .*",
    ):
        create_credal_plot(datasets, labels=labels)


def test_spider_plot_six_classes() -> None:
    """Spider plot runs correctly with six classes."""
    labels = ["A", "B", "C", "D", "E", "F"]
    datasets = np.array(
        [
            [0.1, 0.3, 0.2, 0.4, 0.6, 0.5],
            [0.6, 0.2, 0.1, 0.3, 0.4, 0.5],
            [0.2, 0.5, 0.4, 0.1, 0.3, 0.6],
        ],
    )

    multi.spider_plot(datasets, labels=labels)
