"""One method for the user to call."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.visualization.input_handling import dispatch_plot


def create_credal_plot(input_data: np.ndarray) -> None:
    """One method for the user to call; Refers to correct plotting method via input_handling.

    Args:
    input_data: NumPy array with probabilities.
    """
    dispatch_plot(input_data)
    plt.show()


points_3d_3c = np.array(
    [
        [[0.7, 0.2, 0.1]],
        [[0.4, 0.3, 0.3]],
        [[0.1, 0.8, 0.1]],
        [[0.8, 0.1, 0.1]],
    ],
)

points_2c = np.array(
    [
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
    ],
)

create_credal_plot(points_2c)
