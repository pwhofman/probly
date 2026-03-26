"""Configuration data class for plotting."""

from __future__ import annotations

from dataclasses import dataclass


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
        "#7f8c8d",  # gray
    )

    # Figure / axes defaults
    figure_size: tuple[float, float] = (6.0, 6.0)
    dpi: int = 100

    # Text defaults
    title_fontsize: float = 13.0
    label_fontsize: float = 11.0

    # Primitive styling defaults
    line_width: float = 1.5
    fill_alpha: float = 0.3

    def color(self, index: int) -> str:
        """Return a color from the categorical palette."""
        return self.categorical_palette[index % len(self.categorical_palette)]
