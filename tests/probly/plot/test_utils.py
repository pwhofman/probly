"""Tests for ``probly.plot.utils._plot_line_with_outline``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.plot.utils import _plot_line_with_outline


class TestPlotLineWithOutline:
    """The shared outlined-line helper."""

    def test_returns_main_line_object(self) -> None:
        fig, ax = plt.subplots()
        try:
            x = np.array([0.0, 1.0, 2.0])
            y = np.array([1.0, 0.5, 1.5])
            line = _plot_line_with_outline(ax, x, y, color="red", linewidth=2.0)
            # The main line is the second line drawn (outline first).
            assert line.get_color() == "red"
            assert line.get_linewidth() == 2.0
            # Outline is wider, in white, with a no-legend label.
            outline = ax.lines[0]
            assert outline.get_color() == "white"
            assert outline.get_label() == "_nolegend_"
            assert outline.get_linewidth() > line.get_linewidth()
        finally:
            plt.close(fig)

    def test_uses_axes_color_cycle_when_no_color(self) -> None:
        fig, ax = plt.subplots()
        try:
            line = _plot_line_with_outline(ax, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
            # Should have picked up some non-empty color.
            color = line.get_color()
            assert color is not None
            assert color != ""
        finally:
            plt.close(fig)

    def test_lw_alias_is_consumed(self) -> None:
        fig, ax = plt.subplots()
        try:
            line = _plot_line_with_outline(
                ax,
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                color="blue",
                lw=3.0,
            )
            assert line.get_linewidth() == 3.0
        finally:
            plt.close(fig)
