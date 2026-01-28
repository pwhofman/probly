"""Public API to plot 2d, 3d or multidimensional data as credal sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np  # noqa: TC002

from probly.visualization.credal.input_handling import dispatch_plot

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def create_credal_plot(
    input_data: np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
    choice: str | None = None,
    minmax: bool | None = None,
    *,
    show: bool = True,
) -> Axes:
    """Public API for credal sets; refers to the correct plotting method via input_handling.

    Args:
        input_data: NumPy array with probabilities.
        labels: List of labels corresponding to the input data.
        title: Custom or predefined title.
        choice: Either "MLE", "Credal", "Probability" or None.
        minmax: Enables to show the Min/Max lines only for ternary plots.
        show: Enables the user to decide whether to show the plot or not.
    """
    plot = dispatch_plot(
        input_data,
        labels=labels,
        title=title,
        choice=choice,
        minmax=minmax,
    )

    if show:
        plt.show()

    return plot
