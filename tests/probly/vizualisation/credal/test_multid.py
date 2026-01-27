"""Tests for visualization for more than three classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

mpl.use("Agg")

from matplotlib.collections import PathCollection, PolyCollection
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pytest

from probly.visualization.credal.credal_visualization import create_credal_plot
from probly.visualization.credal.plot_multid import MultiVisualizer

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _spider(
    viz: MultiVisualizer,
    probs: np.ndarray,
    labels: list[str],
    *,
    title: str = "Test",
    mle_flag: bool = True,
    credal_flag: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Helper to call spider_plot with the required signature in this repo."""
    out = viz.spider_plot(probs, labels=labels, title=title, mle_flag=mle_flag, credal_flag=credal_flag, ax=ax)
    return cast("Axes", out)


def test_spider_plot_returns_radar_axes_and_sets_ylim_and_title() -> None:
    viz = MultiVisualizer()
    probs = np.array([[0.2, 0.3, 0.1, 0.4]])
    labels = ["A", "B", "C", "D"]

    ax = _spider(viz, probs, labels, title="My Title", mle_flag=True, credal_flag=True)

    assert ax.name == "radar"
    assert ax.get_ylim()[0] == pytest.approx(0.0)
    assert ax.get_ylim()[1] == pytest.approx(1.0)
    assert ("Spider Plot" in ax.get_title()) or ("My Title" in ax.get_title())

    ax.figure.clf()


def test_spider_plot_adds_expected_legend_entries_when_flags_enabled() -> None:
    viz = MultiVisualizer()
    probs = np.array(
        [
            [0.2, 0.3, 0.1, 0.4],
            [0.1, 0.2, 0.6, 0.1],
        ],
    )
    labels = ["A", "B", "C", "D"]

    ax = _spider(viz, probs, labels, title="Legend", mle_flag=True, credal_flag=True)

    legend = ax.get_legend()
    assert legend is not None
    legend_texts = {t.get_text() for t in legend.get_texts()}

    assert "MLE" in legend_texts
    assert "Credal band (lower-upper)" in legend_texts
    assert "Lower bound" in legend_texts
    assert "Upper bound" in legend_texts

    ax.figure.clf()


def test_spider_plot_mle_scatter_is_at_argmax_of_mean_probs() -> None:
    viz = MultiVisualizer()
    probs = np.array(
        [
            [0.1, 0.1, 0.1, 0.7],
            [0.2, 0.2, 0.1, 0.5],
        ],
    )
    labels = ["A", "B", "C", "D"]

    ax = _spider(viz, probs, labels, title="MLE", mle_flag=True, credal_flag=False)

    n_classes = probs.shape[-1]
    theta = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)

    mean_probs = probs.mean(axis=0)
    max_class = int(np.argmax(mean_probs))

    mle_candidates = [c for c in ax.collections if isinstance(c, PathCollection) and c.get_label() == "MLE"]
    assert len(mle_candidates) == 1

    offsets_raw = mle_candidates[0].get_offsets()
    offsets: NDArray[np.floating[Any]] = np.asarray(offsets_raw, dtype=float)
    assert offsets.shape == (1, 2)

    theta_val = float(offsets[0, 0])
    r_val = float(offsets[0, 1])

    assert theta_val == pytest.approx(theta[max_class])
    assert r_val == pytest.approx(mean_probs[max_class])

    ax.figure.clf()


def test_spider_plot_adds_credal_band_via_fill_between() -> None:
    viz = MultiVisualizer()
    probs = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.1, 0.4, 0.2, 0.6],
        ],
    )
    labels = ["A", "B", "C", "D", "E"]

    ax = _spider(viz, probs, labels, title="Band", mle_flag=False, credal_flag=True)

    poly = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(poly) >= 1

    ax.figure.clf()


def test_spider_plot_adds_lower_and_upper_bound_lines_and_closes_them() -> None:
    viz = MultiVisualizer()
    probs = np.array(
        [
            [0.2, 0.3, 0.1, 0.4],
            [0.1, 0.2, 0.6, 0.1],
        ],
    )
    labels = ["A", "B", "C", "D"]

    ax = _spider(viz, probs, labels, title="Bounds", mle_flag=False, credal_flag=True)

    lower_lines = [ln for ln in ax.lines if ln.get_label() == "Lower bound"]
    upper_lines = [ln for ln in ax.lines if ln.get_label() == "Upper bound"]
    assert len(lower_lines) == 1
    assert len(upper_lines) == 1

    for ln in (lower_lines[0], upper_lines[0]):
        x_raw, y_raw = ln.get_data()
        x = np.asarray(x_raw, dtype=float)
        y = np.asarray(y_raw, dtype=float)
        assert x[0] == pytest.approx(x[-1])
        assert y[0] == pytest.approx(y[-1])

    ax.figure.clf()


def test_spider_plot_raises_if_labels_too_short() -> None:
    viz = MultiVisualizer()
    probs = np.array([[0.2, 0.3, 0.1, 0.4]])
    labels = ["A", "B"]

    with pytest.raises(ValueError, match=r"(ticklabels|FixedLocator|labels|match)"):
        _spider(viz, probs, labels, title="Bad", mle_flag=True, credal_flag=True)


def test_spider_plot_single_dataset_runs() -> None:
    viz = MultiVisualizer()
    labels = ["A", "B", "C", "D"]
    datasets = np.array([[0.1, 0.4, 0.3, 0.2]])

    ax = _spider(viz, datasets, labels, title="Single", mle_flag=True, credal_flag=True)
    assert ax.name == "radar"
    ax.figure.clf()


def test_spider_plot_multiple_datasets_runs() -> None:
    viz = MultiVisualizer()
    labels = ["A", "B", "C", "D", "E"]
    datasets = np.array(
        [
            [0.1, 0.5, 0.2, 0.3, 0.7],
            [0.3, 0.6, 0.1, 0.4, 0.2],
            [0.8, 0.1, 0.05, 0.2, 0.3],
        ],
    )

    ax = _spider(viz, datasets, labels, title="Multi", mle_flag=True, credal_flag=True)
    assert ax.name == "radar"
    ax.figure.clf()


def test_spider_plot_six_classes_runs() -> None:
    viz = MultiVisualizer()
    labels = ["A", "B", "C", "D", "E", "F"]
    datasets = np.array(
        [
            [0.1, 0.3, 0.2, 0.4, 0.6, 0.5],
            [0.6, 0.2, 0.1, 0.3, 0.4, 0.5],
            [0.2, 0.5, 0.4, 0.1, 0.3, 0.6],
        ],
    )

    ax = _spider(viz, datasets, labels, title="Six", mle_flag=True, credal_flag=True)
    assert ax.name == "radar"
    ax.figure.clf()


def test_spider_plot_uses_passed_in_axes() -> None:
    viz = MultiVisualizer()
    probs = np.array([[0.2, 0.3, 0.1, 0.4]])
    labels = ["A", "B", "C", "D"]

    _spider(viz, probs, labels, title="Register", mle_flag=True, credal_flag=True)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "radar"})
    out = _spider(viz, probs, labels, title="Reuse", mle_flag=True, credal_flag=True, ax=ax)

    assert out is ax
    fig.clf()


def test_create_credal_plot_length_mismatch_raises() -> None:
    labels = ["A", "B", "C"]
    datasets = np.array([[0.1, 0.2, 0.3, 0.4]])

    with pytest.raises(ValueError, match="Number of labels"):
        create_credal_plot(datasets, labels=labels)
