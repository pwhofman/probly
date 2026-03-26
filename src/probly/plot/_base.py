"""Base plot functions and protocols for plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .config import PlotConfig

DataT_contra = TypeVar("DataT_contra", contravariant=True)


class PlotFunction(Protocol[DataT_contra]):
    """Protocol for typed, composable plotting functions.

    Implementations must accept the data as the first positional-only argument,
    followed by keyword-only arguments for axes, title, config, and show.
    """

    def __call__(
        self,
        data: DataT_contra,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        config: PlotConfig | None = None,
        show: bool = False,
    ) -> Axes:
        """A callable plotting function.

        Args:
            data: The data to plot (positional-only; implementations may use any name).
            ax: The matplotlib axes to plot on. If None, a new figure and axes will be created.
            title: Title of the plot.
            config: Configuration for plotting.
            show: Whether to call plt.show() after plotting.
        """
        ...
