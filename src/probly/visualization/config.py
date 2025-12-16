"""Central configuration for probly visualizations (colors, styles, etc.)."""

from __future__ import annotations

__all__ = [
    "BLACK",
    "BLUE",
    "LINES",
    "NEUTRAL",
    "RED",
    "get_sign_color",
]

RED: str = "#ff0d57"
BLUE: str = "#1e88e5"
BLACK: str = "#000000"
NEUTRAL: str = "#ffffff"
LINES: str = "#cccccc"
HULL_EDGE: str = BLUE
HULL_FACE: str = BLUE
HULL_FACE_ALPHA: float = 0.25
HULL_EDGE_WIDTH: float = 2.0


def get_sign_color(value: float) -> str:
    """Return RED for >= 0 and BLUE for < 0."""
    return RED if value >= 0 else BLUE
