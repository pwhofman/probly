"""Input Handling here."""

from __future__ import annotations

import numpy as np


def check_num_classes(input_data: np.ndarray) -> int:
    """Checks number of classes and refers to respective function.

    Args:
    input_data: array with last dimension equal to the number of classes.

    Returns:
    Number of classes.
    """
    if input_data.shape[-1] == 2:
        return 2
    if input_data.shape[-1] == 3:
        return 3
    if input_data.shape[-1] == 4:
        return 4
    raise NotImplementedError


def check_shape(input_data: np.ndarray) -> None:
    """Sanity check.

    Args:
    input_data: 3D tensor.
    """
    msg1 = "Input must be a tensor with shape (n_models, n_samples, n_classes."
    msg2 = "There must be at least 2 classes."
    msg3 = "The probabilities of each class must sum to 1."
    msg4 = "All probabilities must be positive."
    if not isinstance(input_data, np.ndarray) and input_data.ndim != 3:
        raise ValueError(msg1)
    if input_data.shape[2] <= 1:
        raise ValueError(msg2)
    if not np.allclose(input_data.sum(axis=-1), 1):
        raise ValueError(msg3)
    if (input_data < 0).any():
        raise ValueError(msg4)
