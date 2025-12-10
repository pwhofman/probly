"""One method for the user to call."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.visualization.input_handling import dispatch_plot


def create_credal_plot(
    input_data: np.ndarray,
    labels: list[str] | None = None,
) -> None:
    """One method for the user to call; Refers to correct plotting method via input_handling.

    Args:
    input_data: NumPy array with probabilities.
    labels: List of labels corresponding to the input data.
    """
    dispatch_plot(input_data, labels=labels)
    plt.show()


points = np.array(
    [
        [0.1, 0.2, 0.7],
        [0.1, 0.1, 0.8],
        [0.2, 0.3, 0.5],
    ],
)
