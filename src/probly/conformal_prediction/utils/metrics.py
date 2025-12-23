"""includes:
empirical coverage, avg.set.size.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def empirical_coverage(
    prediction_sets: npt.NDArray[np.bool_],
    true_labels: Sequence[Any],
) -> float:
    """Compute empirical coverage of prediction sets.

    Args:
        prediction_sets: A (n_instances, n_labels) boolean array indicating
                         which labels are included in each prediction set.
        true_labels: A sequence of true labels for each instance.

    Returns:
        Empirical coverage as a float.
    """
    n_instances = prediction_sets.shape[0]
    correct_inclusion = 0

    for i in range(n_instances):
        true_label = true_labels[i]
        if true_label < prediction_sets.shape[1]:  # assuming labels are 0-indexed
            if prediction_sets[i, true_label]:
                correct_inclusion += 1

    coverage = correct_inclusion / n_instances
    return coverage


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
    return avg_size
