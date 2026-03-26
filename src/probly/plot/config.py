"""Configuration data class for plotting."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl

_FONT_FAMILY = "Fira Sans"


def _apply_rc_defaults() -> None:
    """Set global matplotlib rcParams for the Fira Sans font family."""
    rc = mpl.rcParams
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = [_FONT_FAMILY]

    rc["xtick.labelsize"] = 10
    rc["ytick.labelsize"] = 10

    rc["axes.labelsize"] = 12
    rc["axes.labelweight"] = 600

    rc["axes.titlesize"] = 14
    rc["axes.titleweight"] = 700

    rc["font.weight"] = 300
    rc["legend.fontsize"] = 10
    rc["figure.titlesize"] = 16
    rc["figure.titleweight"] = 700


_apply_rc_defaults()


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Shared configuration for plotting functions.

    Plot functions should accept ``config: PlotConfig | None = None`` and fall
    back to ``PlotConfig()`` when no config is provided.
    """

    # General categorical palette
    categorical_palette: tuple[str, ...] = (
        "#1e88e5",  # blue
        "#ff0d57",  # pink-red
        "#2ecc71",  # green
        "#f39c12",  # orange
        "#9b59b6",  # purple
        "#16a085",  # teal
        "#d35400",  # dark orange
    )
    color_positive: str = "#ff0d57"  # pink-red
    color_negative: str = "#1e88e5"  # blue
    color_neutral: str = "#7f8c8d"  # gray
    color_gridline: str = "#e0e0e0"  # light-gray

    # Figure / axes defaults
    figure_size: tuple[float, float] = (6.0, 6.0)
    dpi: int = 100

    # Text defaults
    title_fontsize: float = 14.0
    label_fontsize: float = 12.0

    # Primitive styling defaults
    line_width: float = 1.5
    fill_alpha: float = 0.3
    marker_size: float = 30.0
    grid_alpha: float = 0.5
    grid_linestyle: str = "--"
    histogram_alpha: float = 0.6

    def color(self, index: int) -> str:
        """Return a color from the categorical palette.

        Args:
            index: Index to select color from the palette. Will wrap around if
                index exceeds palette length.

        Returns:
            A hex color string from the categorical palette.
        """
        return self.categorical_palette[index % len(self.categorical_palette)]
