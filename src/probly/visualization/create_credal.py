"""One method for the user to call."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np  # noqa: TC002

from probly.visualization.input_handling import dispatch_plot


def create_credal_plot(
    input_data: np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
    choice: str | None = None,
    minmax: bool | None = None,
    *,
    show: bool = True,
) -> plt.Axes | None:
    """One method for the user to call; Refers to correct plotting method via input_handling.

    Args:
    input_data: NumPy array with probabilities.
    labels: List of labels corresponding to the input data.
    title: Custom or predefined title.
    choice: Either "MLE", "Credal", "Probability" or None.
    minmax: Enables to show the Min/Max lines for ternary plots.
    show: Enables the user to decide whether to show the plot or not.
    """
    plot = dispatch_plot(input_data, labels=labels, title=title, choice=choice, minmax=minmax)
    if show:
        plt.show()
    return plot
