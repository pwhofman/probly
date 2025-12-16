"""Tests for the common LAC implementation including Iris integration."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from probly.conformal_prediction.lac.common import (
    LAC,
    accretive_completion,
    calculate_local_weights,
    calculate_non_conformity_score,
    calculate_weighted_quantile,
)


class MockModel:
    """A dummy model to simulate sklearn's predict/predict_proba."""

    def predict(self, x: Any) -> Any:  # noqa: ARG002, ANN401
        """Return fixed probabilities for testing."""
        # Output shape is (n_samples, n_classes)
        return np.array(
            [
                [0.1, 0.8, 0.1],
                [0.6, 0.3, 0.1],
                [0.2, 0.2, 0.6],
            ],
        )


def test_calculate_non_conformity_score() -> None:
    """Test calculation of non-conformity scores: 1 - p(y|x)."""
    probas = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.6, 0.3, 0.1],
            [0.2, 0.2, 0.6],
        ],
    )
    y_true = np.array([1, 0, 2])

    scores = calculate_non_conformity_score(probas, y_true)
    expected = np.array([0.2, 0.4, 0.4])
    np.testing.assert_allclose(scores, expected, atol=1e-6)


def test_accretive_completion() -> None:
    """Test that empty sets are filled with the class of highest probability."""
    prediction_sets = np.array(
        [
            [False, False, False],
            [True, False, False],
        ],
    )
    probas = np.array(
        [
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
        ],
    )

    result = accretive_completion(prediction_sets, probas)
    expected = np.array(
        [
            [False, True, False],  # Filled!
            [True, False, False],  # Unchanged
        ],
    )
    np.testing.assert_array_equal(result, expected)


def test_calculate_weighted_quantile() -> None:
    """Test standard quantile calculation."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = calculate_weighted_quantile(values, 0.5, sample_weight=None)
    assert q == 3.0


def test_calculate_local_weights() -> None:
    """Test that weights are returned as ones (default for LAC)."""
    x = np.zeros((5, 3))
    weights = calculate_local_weights(x)
    assert weights.shape == (5,)
    assert np.all(weights == 1.0)


def test_lac_class_integration() -> None:
    """Test the LAC class flow with dummy data."""
    model = MockModel()
    predictor = LAC(model)

    x_dummy = np.zeros((3, 5))
    y_dummy = np.array([1, 0, 2])

    scores = predictor._compute_nonconformity(x_dummy, y_dummy)  # noqa: SLF001

    expected_scores = np.array([0.2, 0.4, 0.4])
    np.testing.assert_allclose(scores, expected_scores, atol=1e-6)

    predictor.is_calibrated = True
    predictor.threshold = 0.5
    res = predictor.predict(x_dummy, significance_level=0.1)

    assert isinstance(res, list)
    assert len(res) == 3
    assert res[0][1]


# REAL WORLD DATASET TEST (IRIS)


class SklearnWrapper:
    """Wraps scikit-learn model to return probabilities on .predict()."""

    def __init__(self, model: Any) -> None:  # noqa: ANN401
        """Initialize the wrapper with a model."""
        self.model = model

    def predict(self, x: Any) -> Any:  # noqa: ANN401
        """Redirect predict to predict_proba."""
        return self.model.predict_proba(x)


def test_iris_common_integration() -> None:
    """Test LAC on the Iris dataset using Sklearn LogisticRegression."""
    # 1. Load Data
    data = load_iris()
    x, y = data.data, data.target

    # Ensure y is integer array
    y = y.astype(int)

    # 2. Split: Train (40%), Calibrate (40%), Test (20%)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        train_size=0.4,
        random_state=42,
    )
    x_cal, x_test, y_cal, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.33,
        random_state=42,
    )

    # 3. Train Base Model
    # Multi_class='multinomial' ensures we get (n_samples, n_classes) probas
    base_model = LogisticRegression(max_iter=200, multi_class="multinomial")
    base_model.fit(x_train, y_train)

    # WRAP IT: LAC expects .predict() to return probabilities
    model = SklearnWrapper(base_model)

    # 4. Calibrate LAC
    predictor = LAC(model)

    # Simulate manual calibration (calculate scores on cal set)
    cal_probas = model.predict(x_cal)
    cal_scores = calculate_non_conformity_score(cal_probas, y_cal)

    # Target 90% coverage -> alpha = 0.1
    alpha = 0.1
    q_val = calculate_weighted_quantile(cal_scores, quantile=1.0 - alpha)

    predictor.is_calibrated = True
    predictor.threshold = q_val

    # 5. Predict on Test Set
    prediction_sets = predictor.predict(x_test, significance_level=alpha)

    # 6. Check Metrics
    covered_count = 0
    set_sizes = []
    empty_count = 0

    for i, p_set in enumerate(prediction_sets):
        true_label = y_test[i]
        set_size = np.sum(p_set)
        set_sizes.append(set_size)

        if set_size == 0:
            empty_count += 1

        # Check coverage
        if p_set[true_label]:
            covered_count += 1

    coverage = covered_count / len(y_test)
    avg_size = np.mean(set_sizes)
    empty_rate = empty_count / len(y_test)

    # 7. Assertions
    assert coverage >= 0.8, f"Coverage too low: {coverage:.2f}"
    assert 1.0 <= avg_size <= 3.0, f"Avg set size weird: {avg_size:.2f}"
    assert empty_rate == 0.0, "Found empty prediction sets!"
