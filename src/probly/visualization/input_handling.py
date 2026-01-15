"""Input Handling here."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from probly.visualization.plot_2d import IntervalVisualizer
from probly.visualization.plot_3d import TernaryVisualizer
from probly.visualization.plot_multid import MultiVisualizer


def check_num_classes(input_data: np.ndarray) -> int:
    """Checks number of classes and refers to respective function.

    Args:
    input_data: array with last dimension equal to the number of classes.

    Returns:
    Number of classes.
    """
    n_classes = int(input_data.shape[-1])
    return n_classes


def check_shape(input_data: np.ndarray) -> np.ndarray:
    """Sanity check.

    Args:
    input_data: Minimum 2D NumPy array with probability vector.

    Returns:
    input_data or Error Message.
    """
    msg1 = "Input must be a NumPy Array."
    msg2 = "Input must not be empty."
    msg3 = "Input must be at least 2D."
    msg4 = "The probabilities of each class must sum to 1."
    msg5 = "All probabilities must be positive."

    # Validates that input_data is either a 2D or 3D NumPy Array.
    if not isinstance(input_data, np.ndarray):
        raise TypeError(msg1)
    if input_data.size == 0:
        raise ValueError(msg2)
    if input_data.ndim < 2:
        raise ValueError(msg3)

    # Validates the input probabilities.
    if not np.allclose(input_data.sum(axis=-1), 1):
        raise ValueError(msg4)
    if (input_data < 0).any():
        raise ValueError(msg5)
    return input_data


def normalize_input(input_data: np.ndarray) -> np.ndarray:
    """Normalize input data.

    Args:
        input_data: array with last dimension equal to the number of classes.

    Returns:
    2D NumPy array with normalized input data.
    """
    if input_data.ndim >= 3:
        input_data = input_data.reshape(-1, input_data.shape[-1])
        return input_data
    return input_data


def dispatch_plot(
    input_data: np.ndarray,
    labels: list[str] | None = None,
) -> plt.Axes:
    """Selects and executes the correct plotting function based on class count.

    Args:
        input_data: Probabilities vector.
        labels: List of labels corresponding to the classes.
    """
    # Validates input.
    input_data = check_shape(input_data)

    # Normalizes input.
    points = normalize_input(input_data)

    # Counts the number of classes to refer correctly
    n_classes = check_num_classes(points)

    if labels is None:
        labels = [f"C{i + 1}" for i in range(n_classes)]

    if len(labels) != n_classes:
        msg = f"Number of labels ({len(labels)}) must match number of classes ({n_classes})."
        raise ValueError(msg)

    # Depending on number of classes chooses correct plotting function.
    if n_classes == 2:
        viz = IntervalVisualizer()
        return viz.interval_plot(points, labels=labels)

    if n_classes == 3:
        ter = TernaryVisualizer()
        ax = ter.ternary_plot(points, labels=labels)
        return ax

    multi = MultiVisualizer()
    return multi.spider_plot(points, labels=labels)
