"""Utility functions for evaluating conformal prediction metrics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def empirical_coverage(
    prediction_sets: npt.NDArray[np.bool_],
    true_labels: npt.NDArray[np.integer],
) -> float:
    """Compute empirical coverage of prediction sets.

    Args:
        prediction_sets: A (n_instances, n_labels) boolean array indicating
                         which labels are included in each prediction set.
        true_labels: A sequence of true labels for each instance.

    Returns:
        Empirical coverage as a float.

    Raises:
        ValueError: If the number of instances in prediction_sets and true_labels do not match.
    """
    n_instances = prediction_sets.shape[0]
    if len(true_labels) != n_instances:
        msg = f"Shape mismatch: prediction_sets has {n_instances} instances but true_labels has {len(true_labels)}"
        raise ValueError(msg)
    correct_inclusion = 0

    for i in range(n_instances):
        true_label = int(true_labels[i])
        if 0 <= true_label < prediction_sets.shape[1] and prediction_sets[i, true_label]:
            correct_inclusion += 1  # assuming labels are 0-indexed

    coverage = correct_inclusion / n_instances
    return float(coverage)


def average_set_size(
    prediction_sets: npt.NDArray[np.bool_],
) -> float:
    """Compute average size of prediction sets.

    Args:
        prediction_sets: A (n_instances, n_labels) boolean array indicating
                         which labels are included in each prediction set.

    Returns:
        Average size of the prediction sets as a float.
    """
    n_instances = prediction_sets.shape[0]
    total_size = np.sum(prediction_sets)
    avg_size = total_size / n_instances
    return float(avg_size)
