"""Tests for Accretive completion LAC method."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probly.conformal_prediction.lac.common import accretive_completion


def test_fills_empty_set_with_max_score() -> None:
    """Scenario: A prediction set is completely empty (all False).

    Expectation: The method must force the class with the highest score (0.8 -> Index 1) to True.
    """
    # Input: 1 Sample, 2 Classes. Both False.
    pred_set = np.array([[False, False]])
    # Scores: Class 0 = 0.2, Class 1 = 0.8
    scores = np.array([[0.2, 0.8]])

    result = accretive_completion(pred_set, scores)

    # 1. Verify that the set is no longer empty
    assert np.any(result), "The set should not be empty anymore."

    # 2. Verify that exactly [False, True] was selected
    expected = np.array([[False, True]])
    np.testing.assert_array_equal(result, expected, err_msg="The class with the highest score was not selected.")


def test_leaves_non_empty_set_untouched() -> None:
    """Scenario: A set is already filled/valid (contains at least one True).

    Expectation: The method must NOT modify the set, even if another class has a higher score.
    """
    # Input: Class 0 is True.
    pred_set = np.array([[True, False]])
    # Scores: Class 1 has a higher score (0.9),
    # but since the set is not empty, it should remain unchanged.
    scores = np.array([[0.1, 0.9]])

    result = accretive_completion(pred_set, scores)

    np.testing.assert_array_equal(result, pred_set, err_msg="A valid (non-empty) set must not be modified.")


def test_mixed_batch() -> None:
    """Scenario: Processing a batch with mixed cases (one empty, one valid).

    Row 0: Empty -> Must be repaired.
    Row 1: Full -> Must remain unchanged.
    """
    pred_sets = np.array(
        [
            [False, False],  # Empty -> needs repair
            [True, False],  # Full -> should stay as is
        ],
    )
    scores = np.array(
        [
            [0.7, 0.3],  # Row 0: Index 0 has highest score
            [0.4, 0.6],  # Row 1: Scores don't matter here
        ],
    )

    expected = np.array(
        [
            [True, False],  # Repaired: Index 0 forced to True
            [True, False],  # Unchanged
        ],
    )

    result = accretive_completion(pred_sets, scores)
    np.testing.assert_array_equal(result, expected)


def test_randomized_large_batch() -> None:
    """Stress-Test: Generates random data to simulate a 'real' dataset scenario.

    Ensures that:
    1. Accretive completion handles large batches (100 samples) correctly.
    2. Empty sets are filled with the class of maximum probability.
    3. Valid sets remain strictly untouched.
    """
    rng = np.random.default_rng(42)
    n_samples = 100
    n_classes = 10

    # 1. Generate Random Scores (Probabilities)
    scores = rng.random((n_samples, n_classes))

    # 2. Generate Prediction Sets
    # Create sets where ~30% are empty/False initially to simulate high uncertainty
    pred_sets = rng.choice([True, False], size=(n_samples, n_classes), p=[0.3, 0.7])

    # Manually force the first 10 rows to be completely empty to GUARANTEE edge cases
    pred_sets[0:10, :] = False

    # Keep a copy for verification
    original_sets = pred_sets.copy()

    # Execute
    result = accretive_completion(pred_sets, scores)

    # CHECK 1: Global Admissibility (No empty sets allowed)
    row_sums = np.sum(result, axis=1)
    assert np.all(row_sums > 0), "Found empty sets after completion! Admissibility violated."

    # CHECK 2: Element-wise Verification
    for i in range(n_samples):
        original_row = original_sets[i]
        new_row = result[i]

        is_originally_empty = np.sum(original_row) == 0

        if is_originally_empty:
            # Case A: Row was empty -> Must now contain exactly the max-score class
            expected_idx = np.argmax(scores[i])

            assert new_row[expected_idx], f"Row {i}: Failed to add the class with max score."
            assert np.sum(new_row) == 1, f"Row {i}: Added more than one class to an empty set."
        else:
            # Case B: Row was valid -> Must be identical to original
            np.testing.assert_array_equal(
                new_row,
                original_row,
                err_msg=f"Row {i}: Modified a valid set! Valid sets must be preserved.",
            )


def test_with_iris_probabilities() -> None:
    """Integration Scenario: Use REAL probability distributions from Iris.

    This verifies logic on sharp/realistic probability distributions
    rather than uniform random noise.
    """
    # 1. Prepare Real Data
    iris = load_iris()
    x, y = iris.data, iris.target

    # Standardize to ensure LogisticRegression converges cleanly
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Train a simple classifier to get realistic probabilities
    clf = LogisticRegression(random_state=42)
    clf.fit(x_scaled, y)

    # Get scores (probabilities) for all samples
    # The shape is (150, 3)
    scores = clf.predict_proba(x_scaled)

    # 2. Simulate "Worst Case" Calibration (Threshold = 1.0)
    # We pass in a fully empty boolean matrix.
    # This forces accretive_completion to pick the best class for EVERY sample.
    empty_sets = np.zeros_like(scores, dtype=bool)

    # 3. Execute
    result = accretive_completion(empty_sets, scores)

    # 4. Verify
    # Since we started with empty sets, the result should be exactly the argmax class.
    # This effectively turns the Conformal Predictor into a standard Classifier.
    predicted_classes = np.argmax(result, axis=1)
    highest_score_classes = np.argmax(scores, axis=1)

    np.testing.assert_array_equal(
        predicted_classes,
        highest_score_classes,
        err_msg="On Iris data, completion did not select the class with highest probability.",
    )

    # Ensure strictly 1 class per row (since we started with empty)
    set_sizes = np.sum(result, axis=1)
    assert np.all(set_sizes == 1), "Should have exactly 1 class per row for initially empty sets."
