"""Configuration data class for plotting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Shared configuration for plotting functions.

    This config stores library-wide defaults for colors, sizing, and styling.
    Plot functions should accept ``config: PlotConfig | None = None`` and fall
    back to ``PlotConfig()`` when no config is provided.
    """

    # Semantic colors
    primary_color: str = "#1e88e5"
    secondary_color: str = "#ff0d57"
    positive_color: str = "#2ecc71"
    negative_color: str = "#e74c3c"
    neutral_color: str = "#7f8c8d"
    reference_color: str = "#34495e"

    # General categorical palette
    categorical_palette: tuple[str, ...] = (
        "#1e88e5",  # blue
        "#ff0d57",  # pink-red
        "#2ecc71",  # green
        "#f39c12",  # orange
        "#9b59b6",  # purple
        "#16a085",  # teal
        "#d35400",  # dark orange
        "#7f8c8d",  # gray
    )

    # Figure / axes defaults
    figure_size: tuple[float, float] = (6.0, 4.0)
    dpi: int = 100

    # Text defaults
    title_fontsize: float = 13.0
    label_fontsize: float = 11.0
    tick_fontsize: float = 10.0
    legend_fontsize: float = 10.0

    # Primitive styling defaults
    line_width: float = 2.0
    marker_size: float = 6.0
    bar_alpha: float = 0.9
    fill_alpha: float = 0.2

    # Grid / frame defaults
    show_grid: bool = True
    grid_alpha: float = 0.25
    grid_line_style: str = "--"

    # Layout / rendering behavior
    use_tight_layout: bool = True

    def color(self, index: int) -> str:
        """Return a color from the categorical palette."""
        return self.categorical_palette[index % len(self.categorical_palette)]
