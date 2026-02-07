"""Tests for metrics utilities."""

from __future__ import annotations

import numpy as np
import pytest

from probly.conformal_prediction.utils.metrics import (
    average_set_size,
    empirical_coverage,
)


@pytest.fixture
def setup_basic_data() -> tuple[np.ndarray, np.ndarray]:
    """Create basic test data."""
    prediction_sets = np.array(
        [
            [True, True, False, False],  # classes 0,1
            [True, False, False, False],  # classes 0
            [False, True, True, False],  # classes 1,2
            [False, False, True, True],  # classes 2,3
        ],
        dtype=bool,
    )
    labels = np.array([0, 0, 1, 2])
    return prediction_sets, labels


def test_empirical_coverage_basic(setup_basic_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test basic empirical coverage calculation."""
    prediction_sets, labels = setup_basic_data
    coverage = empirical_coverage(prediction_sets, labels)

    assert isinstance(coverage, float)
    assert 0 <= coverage <= 1
    # in this example: 4/4 samples should be covered
    assert coverage == 1.0


def test_empirical_coverage_partial() -> None:
    """Test empirical coverage with partial coverage."""
    prediction_sets = np.array(
        [
            [True, False, False],  # true label 0 - covered
            [False, True, False],  # true label 1 - covered
            [False, False, True],  # true label 2 - not in set (should be False)
        ],
        dtype=bool,
    )
    labels = np.array([0, 1, 0])  # last sample true label is 0, but set has 2

    coverage = empirical_coverage(prediction_sets, labels)
    assert coverage == 2 / 3  # 2 out of 3 covered


def test_empirical_coverage_edge_cases() -> None:
    """Test edge cases for empirical coverage."""
    # empty prediction sets
    prediction_sets = np.array(
        [
            [False, False],
            [False, False],
        ],
        dtype=bool,
    )
    labels = np.array([0, 1])
    coverage = empirical_coverage(prediction_sets, labels)
    assert coverage == 0.0

    # all labels covered
    prediction_sets = np.array(
        [
            [True, True],
            [True, True],
        ],
        dtype=bool,
    )
    coverage = empirical_coverage(prediction_sets, labels)
    assert coverage == 1.0


def test_average_set_size_basic(setup_basic_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test basic average set size calculation."""
    prediction_sets, _ = setup_basic_data
    avg_size = average_set_size(prediction_sets)

    assert isinstance(avg_size, float)
    assert avg_size > 0
    # in this example: sizes are 2,1,2,2 -> average is 1.75
    assert avg_size == 1.75


def test_average_set_size_edge_cases() -> None:
    """Test edge cases for average set size."""
    # empty sets
    prediction_sets = np.array(
        [
            [False, False],
            [False, False],
        ],
        dtype=bool,
    )
    avg_size = average_set_size(prediction_sets)
    assert avg_size == 0.0

    # single class sets
    prediction_sets = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )
    avg_size = average_set_size(prediction_sets)
    assert avg_size == 1.0

    # all classes included
    prediction_sets = np.array(
        [
            [True, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    avg_size = average_set_size(prediction_sets)
    assert avg_size == 3.0


def test_metrics_with_realistic_data() -> None:
    """Test metrics with realistic conformal prediction data."""
    n_samples = 100
    n_classes = 10

    # generate realistic prediction sets
    rng = np.random.default_rng(42)
    prediction_sets = np.zeros((n_samples, n_classes), dtype=bool)

    for i in range(n_samples):
        # random number of classes (1-3)
        n_included = rng.integers(1, 4)
        # random classes
        included_classes = rng.choice(n_classes, size=n_included, replace=False)
        prediction_sets[i, included_classes] = True

    # generate random true labels
    labels = rng.integers(0, n_classes, size=n_samples)

    # calculate metrics
    coverage = empirical_coverage(prediction_sets, labels)
    avg_size = average_set_size(prediction_sets)

    # basic assertions
    assert 0 <= coverage <= 1
    assert 1 <= avg_size <= 3  # should match generation range
    assert isinstance(coverage, float)
    assert isinstance(avg_size, float)


def test_metrics_shape_validation() -> None:
    """Test that shape mismatches are caught."""
    prediction_sets = np.array([[True, False], [False, True]], dtype=bool)
    labels_wrong = np.array([0, 1, 2])  # wrong shape

    # empirical coverage should raise an error on shape mismatch
    with pytest.raises(
        ValueError,
        match=(
            f"Shape mismatch: prediction_sets has {prediction_sets.shape[0]} "
            f"instances but true_labels has {len(labels_wrong)}"
        ),
    ):
        empirical_coverage(prediction_sets, labels_wrong)
