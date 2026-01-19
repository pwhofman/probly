"""Tests for the plot module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from probly.plot.credal import credal_set_plot, simplex_plot


def test_simplex_plot_outputs() -> None:
    probs = np.array([[1 / 3, 1 / 3, 1 / 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    fig, ax = simplex_plot(probs)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.name == "ternary"
    assert ax.collections[0].get_offsets().shape[0] == len(probs)


def test_credal_set_plot() -> None:
    probs = np.array([[1 / 3, 1 / 3, 1 / 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    fig, ax = credal_set_plot(probs)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.name == "ternary"
    assert ax.collections[0].get_offsets().shape[0] == len(probs)

    probs = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
    with pytest.raises(
        ValueError,
        match=r"The set of vertices is empty. Please check the probabilities in the credal set.",
    ):
        credal_set_plot(probs)
