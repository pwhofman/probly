"""Central configuration for probly visualizations (colors, styles, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.colors import Colormap, LinearSegmentedColormap

if TYPE_CHECKING:
    from matplotlib.colorbar import Colorbar

__all__ = [
    "BLUE",
    "CLASS_LABLES_FONTSIZE",
    "COLORBAR_LABEL",
    "COLORBAR_LABEL_FONTSIZE",
    "COLORBAR_LABEL_LOC",
    "COLORBAR_LABEL_PAD",
    "COLORBAR_LABEL_ROTATION",
    "COLORBAR_TICKS",
    "FILL_ALPHA",
    "HULL_EDGE",
    "HULL_EDGE_WIDTH",
    "HULL_FACE",
    "HULL_LINE_WIDTH",
    "LINES",
    "MIN_MAX_ALPHA",
    "MIN_MAX_LINESTYLE_1",
    "MIN_MAX_LINESTYLE_2",
    "MIN_MAX_LINE_WIDTH",
    "PROBLY_CMAP",
    "PROBLY_CMAP_COLORS",
    "PROB_ALPHA",
    "PROB_FONT_SIZE",
    "PROB_LINESTYLE",
    "PROB_LINE_WIDTH",
    "RED",
    "WHITE",
    "_MIN_MAX_STYLES",
    "get_sign_color",
    "style_colorbar",
]

RED: str = "#ff0d57"
BLUE: str = "#1e88e5"
BLACK: str = "#000000"
WHITE: str = "#ffffff"
LINES: str = "#cccccc"
FILL_ALPHA: float = 0.25
CLASS_LABLES_FONTSIZE: int = 8
HULL_LINE_WIDTH: float = 2
HULL_EDGE: str = BLUE
HULL_FACE: str = BLUE
HULL_EDGE_WIDTH: float = 2.0
PROB_LINESTYLE: str = ".."
PROB_LINE_WIDTH: float = 1
PROB_FONT_SIZE: int = 9
PROB_ALPHA: float = 1.0
MIN_MAX_LINE_WIDTH: float = 1.5
MIN_MAX_ALPHA: float = 0.7
MIN_MAX_LINESTYLE_1: str = "--"
MIN_MAX_LINESTYLE_2: str = "-."

_MIN_MAX_STYLES = {
    1: (RED, "--"),
    2: (BLUE, "-."),
}
PROBLY_CMAP_COLORS: tuple[str, str] = (BLUE, RED)
PROBLY_CMAP: Colormap = LinearSegmentedColormap.from_list("probly_colors", PROBLY_CMAP_COLORS)
COLORBAR_LABEL: str = "Uncertainty:"
COLORBAR_LABEL_LOC: str = "left"
COLORBAR_LABEL_ROTATION: float = 90
COLORBAR_LABEL_PAD: float = 10
COLORBAR_LABEL_FONTSIZE: int = 11

COLORBAR_TICKS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def style_colorbar(
    cbar: Colorbar,
    *,
    ticks: tuple[float, ...] = COLORBAR_TICKS,
    label: str = COLORBAR_LABEL,
) -> None:
    """Apply probly default style to a matplotlib colorbar."""
    cbar.set_ticks(ticks)
    cbar.set_label(
        label, rotation=COLORBAR_LABEL_ROTATION, labelpad=COLORBAR_LABEL_PAD, fontsize=COLORBAR_LABEL_FONTSIZE
    )
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.label.set_verticalalignment("center")


def choose_min_max_style(value: int) -> tuple[str, str]:
    try:
        return _MIN_MAX_STYLES[value]
    except KeyError as err:
        msg = f"Unknown min/max style key: {value}"
        raise ValueError(msg) from err


def get_sign_color(value: float) -> str:
    """Return RED for >= 0 and BLUE for < 0."""
    return RED if value >= 0 else BLUE
