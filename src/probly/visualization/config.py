"""Central configuration for probly visualizations (colors, styles, etc.)."""

from __future__ import annotations

__all__ = [
    "BLUE",
    "CLASS_LABLES_FONTSIZE",
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
    "PROB_ALPHA",
    "PROB_FONT_SIZE",
    "PROB_LINESTYLE",
    "PROB_LINE_WIDTH",
    "RED",
    "WHITE",
    "get_sign_color",
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


def get_sign_color(value: float) -> str:
    """Return RED for >= 0 and BLUE for < 0."""
    return RED if value >= 0 else BLUE
