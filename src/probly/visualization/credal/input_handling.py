"""Manages input as well as dispatch to correct plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from probly.visualization.credal.plot_2d import IntervalVisualizer
from probly.visualization.credal.plot_3d import TernaryVisualizer
from probly.visualization.credal.plot_multid import MultiVisualizer


def check_num_classes(input_data: np.ndarray) -> int:
    """Checks number of classes.

    Args:
        input_data: Array with last dimension equal to the number of classes.

    Returns:
        Number of classes.
    """
    n_classes = int(input_data.shape[-1])
    return n_classes


def check_shape(input_data: np.ndarray) -> np.ndarray:
    """Sanity check for input data.

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
    msg6 = "Input must have more than one class."

    if not isinstance(input_data, np.ndarray):
        raise TypeError(msg1)
    if input_data.size == 0:
        raise ValueError(msg2)
    if input_data.ndim < 2:
        raise ValueError(msg3)

    if not np.allclose(input_data.sum(axis=-1), 1):
        raise ValueError(msg4)
    if (input_data < 0).any():
        raise ValueError(msg5)

    if input_data.shape[-1] < 2:
        raise ValueError(msg6)

    return input_data


def normalize_input(input_data: np.ndarray) -> np.ndarray:
    """Normalizes input data.

    Args:
        input_data: Array with last dimension equal to the number of classes.

    Returns:
        2D NumPy array with normalized input data.
    """
    if input_data.ndim >= 3:
        input_data = input_data.reshape(-1, input_data.shape[-1])
        return input_data
    return input_data


def _choice_flag_result(
    choice: str | None = None,
) -> tuple[bool, bool]:
    """Helper function to evaluate the user's choice what to show.

    Args:
        choice: Input String; should be either "MLE", "Credal", "Probability" or None.

    Returns:
        mle_flag, credal_flag as tuple[bool, bool].
    """
    match choice:
        case None:
            mle_flag = True
            credal_flag = True
        case "MLE":
            mle_flag = True
            credal_flag = False
        case "Credal":
            mle_flag = False
            credal_flag = True
        case "Probability":
            mle_flag = False
            credal_flag = False
        case _:
            msgchoice = "Choice must be MLE, Credal, Probability or None."
            raise ValueError(msgchoice)
    return mle_flag, credal_flag


def dispatch_plot(
    input_data: np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
    choice: str | None = None,
    minmax: bool | None = None,
) -> Axes:
    """Selects and executes the correct plotting function based on class count.

    Args:
        input_data: Probabilities vector.
        labels: List of labels corresponding to the classes.
        title: Manages custom or predefined title.
        choice: Allows either "MLE", "Credal", "Probability" or None.
        minmax: Enables to show the Min/Max lines for ternary plots.
    """
    input_data = check_shape(input_data)

    points = normalize_input(input_data)

    n_classes = check_num_classes(points)

    if labels is None:
        labels = [f"C{i + 1}" for i in range(n_classes)]

    if len(labels) != n_classes:
        msg = f"Number of labels ({len(labels)}) must match number of classes ({n_classes})."
        raise ValueError(msg)

    if title is None:
        title = f"Credal Plot ({n_classes} Classes)"

    mle_flag, credal_flag = _choice_flag_result(choice)

    if minmax is None or (minmax is True and credal_flag is False):
        minmax = False

    if n_classes == 2:
        viz = IntervalVisualizer()
        return viz.interval_plot(points, labels=labels, title=title, mle_flag=mle_flag, credal_flag=credal_flag)

    if n_classes == 3:
        ter = TernaryVisualizer()
        ax = ter.ternary_plot(
            points,
            labels=labels,
            title=title,
            mle_flag=mle_flag,
            credal_flag=credal_flag,
            minmax_flag=minmax,
        )
        return ax

    multi = MultiVisualizer()
    return multi.spider_plot(points, labels=labels, title=title, mle_flag=mle_flag, credal_flag=credal_flag)
