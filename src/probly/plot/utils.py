"""Shared plotting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import rcParams

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    import numpy as np


def _plot_line_with_outline(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    outline_color: str = "white",
    outline_delta: float = 2.0,
    **kwargs: object,
) -> Line2D:
    """Plot a line with a thicker white outline for visibility.

    The outline is drawn first so the main line appears on top. Keyword
    arguments are forwarded to ``Axes.plot`` for the main line.

    Args:
        ax: Axes to plot on.
        x: X coordinates of the line.
        y: Y coordinates of the line.
        outline_color: Color of the outline line.
        outline_delta: Extra line width added to the outline.
        **kwargs: Keyword arguments forwarded to ``Axes.plot``.

    Returns:
        The ``Line2D`` for the main line.
    """
    base_lw: float = kwargs.pop("linewidth", kwargs.pop("lw", rcParams["lines.linewidth"]))  # ty: ignore[invalid-assignment]
    color = kwargs.get("color", kwargs.get("c"))
    if color is None:
        color = ax._get_lines.get_next_color()  # ty: ignore[unresolved-attribute]  # noqa: SLF001
    kwargs["color"] = color
    kwargs["linewidth"] = base_lw
    kwargs.pop("lw", None)

    outline_kwargs = dict(kwargs)
    outline_kwargs["color"] = outline_color
    outline_kwargs["linewidth"] = base_lw + outline_delta
    outline_kwargs["label"] = "_nolegend_"

    ax.plot(x, y, **outline_kwargs)  # ty: ignore[invalid-argument-type]
    (line,) = ax.plot(x, y, **kwargs)  # ty: ignore[invalid-argument-type]

    return line
