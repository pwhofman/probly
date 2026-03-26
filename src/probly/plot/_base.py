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
    followed by keyword-only arguments for title, config, and show. Each
    implementation is responsible for creating its own figure and axes internally.
    """

    def __call__(
        self,
        data: DataT_contra,
        *,
        title: str | None = None,
        config: PlotConfig | None = None,
        show: bool = False,
    ) -> Axes:
        """A callable plotting function.

        Args:
            data: The data to plot (positional-only; implementations may use any name).
            title: Title of the plot.
            config: Configuration for plotting.
            show: Whether to call plt.show() after plotting.
        """
        ...
