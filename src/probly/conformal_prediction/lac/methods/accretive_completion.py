"""Accretive completion method for LAC."""

from __future__ import annotations

import numpy as np


def accretive_completion(prediction_sets: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Implements Accretive Completion to eliminate empty prediction sets (Null Regions).

    Based on Sadinle et al. (2019), "Least Ambiguous Set-Valued Classifiers With Bounded Error Levels".
    If a prediction set is empty (all classes are False), this function forces the inclusion
    of the class with the highest score (probability) for that instance.
    This corresponds to lowering the threshold locally until the set is non-empty.

    Args:
        prediction_sets (np.ndarray): Boolean array of shape (n_samples, n_classes).
                                      True indicates the class is in the set.
        scores (np.ndarray): Array of shape (n_samples, n_classes).
                             Usually conditional probabilities p(y|x).
                             High score implies higher likelihood of the class.

    Returns:
        np.ndarray: The modified prediction sets where every row has at least one True.
    """
    # 1. Create a copy to avoid modifying the input array in-place
    completed_sets = prediction_sets.copy()

    # 2. Identify rows that are empty (sum of boolean values in the row is 0)
    # np.sum over axis 1 counts how many True values are in each row
    set_sizes = np.sum(completed_sets, axis=1)
    empty_rows_mask = set_sizes == 0

    # Check if there are any empty sets to process
    if not np.any(empty_rows_mask):
        return completed_sets

    # 3. For the empty rows, find the index of the class with the maximum score
    # scores[empty_rows_mask] selects only the problematic rows
    # np.argmax returns the index of the highest value in those rows
    best_class_indices = np.argmax(scores[empty_rows_mask], axis=1)

    # 4. Get the row indices of the empty sets
    row_indices = np.where(empty_rows_mask)[0]

    # 5. Force the best class to True for these rows
    # This uses numpy advanced indexing: completed_sets[row, col] = True
    completed_sets[row_indices, best_class_indices] = True

    return completed_sets
