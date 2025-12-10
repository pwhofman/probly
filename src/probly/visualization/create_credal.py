"""One method for the user to call."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
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
