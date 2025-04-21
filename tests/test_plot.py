"""Tests for the plot module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.plot import simplex_plot


def test_simplex_plot_outputs():
    probs = np.array([[1 / 3, 1 / 3, 1 / 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    fig, ax = simplex_plot(probs)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.name == "ternary"
    assert ax.collections[0].get_offsets().shape[0] == len(probs)
