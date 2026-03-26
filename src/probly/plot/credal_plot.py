"""Credal plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from probly.representation.credal_set.array import ArrayProbabilityIntervalsCredalSet

    from .config import PlotConfig

__all__ = ["plot_credal_set"]


def plot_credal_set(
    data: ArrayProbabilityIntervalsCredalSet,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    config: PlotConfig | None = None,
    show: bool = False,
) -> Axes:
    """Plot a credal set."""
    msg = "Plotting of credal sets is not implemented yet. This function is a placeholder for future implementation."
    raise NotImplementedError(msg)
