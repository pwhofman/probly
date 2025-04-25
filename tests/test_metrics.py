"""Tests for the metrics module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from probly.metrics import (
    brier_score,
    coverage,
    coverage_convex_hull,
    covered_efficiency,
    efficiency,
    expected_calibration_error,
    log_loss,
    spherical_score,
    zero_one_loss,
)
from tests.utils import validate_uncertainty


@pytest.fixture
def sample_zero_order_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    probs = rng.dirichlet(np.ones(3), 10)
    targets = rng.integers(0, 3, 10)
    return probs, targets


@pytest.fixture
def sample_first_order_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    probs = rng.dirichlet(np.ones(3), (10, 5))
    targets = rng.dirichlet(np.ones(3), 10)
    return probs, targets


@pytest.fixture
def sample_conformal_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    probs = rng.choice([True, False], (10, 3))
    targets = rng.integers(0, 3, 10)
    return probs, targets


def validate_metric(metric: float) -> None:
    assert isinstance(metric, float)
    assert not math.isnan(metric)
    assert not math.isinf(metric)
    assert metric >= 0


def test_expected_calibration_error() -> None:
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    targets = np.array([0, 1, 2, 0])
    ece = expected_calibration_error(probs, targets)
    validate_metric(ece)
    assert ece == 0.0

    targets = np.array([1, 2, 0, 1])
    ece = expected_calibration_error(probs, targets)
    validate_metric(ece)
    assert ece == 1.0


def test_coverage(
    sample_conformal_data: tuple[np.array, np.array], sample_first_order_data: tuple[np.array, np.array]
) -> None:
    preds, targets = sample_conformal_data
    cov = coverage(preds, targets)
    validate_metric(cov)

    probs, targets = sample_first_order_data
    cov = coverage(probs, targets)
    validate_metric(cov)


def test_efficiency(
    sample_conformal_data: tuple[np.array, np.array], sample_first_order_data: tuple[np.array, np.array]
) -> None:
    preds, _ = sample_conformal_data
    eff = efficiency(preds)
    validate_metric(eff)

    probs, _ = sample_first_order_data
    eff = efficiency(probs)
    validate_metric(eff)


def test_coverage_convex_hull(sample_first_order_data: tuple[np.array, np.array]) -> None:
    probs, targets = sample_first_order_data
    cov = coverage_convex_hull(probs, targets)
    validate_metric(cov)


def test_covered_efficiency(
    sample_conformal_data: tuple[np.array, np.array], sample_first_order_data: tuple[np.array, np.array]
) -> None:
    preds, targets = sample_conformal_data
    eff = covered_efficiency(preds, targets)
    validate_metric(eff)

    probs, targets = sample_first_order_data
    eff = covered_efficiency(probs, targets)
    validate_metric(eff)


def test_log_loss(
    sample_zero_order_data: tuple[np.array, np.array], sample_first_order_data: tuple[np.array, np.array]
) -> None:
    probs, targets = sample_zero_order_data
    loss = log_loss(probs, targets)
    validate_metric(loss)

    probs, _ = sample_first_order_data
    loss = log_loss(probs, None)
    validate_uncertainty(loss)


def test_brier_score(
    sample_zero_order_data: tuple[np.array, np.array], sample_first_order_data: tuple[np.array, np.array]
) -> None:
    probs, targets = sample_zero_order_data
    loss = brier_score(probs, targets)
    validate_metric(loss)

    probs, _ = sample_first_order_data
    loss = brier_score(probs, None)
    validate_uncertainty(loss)


def test_zero_one_loss(
    sample_zero_order_data: tuple[np.array, np.array], sample_first_order_data: tuple[np.array, np.array]
) -> None:
    probs, targets = sample_zero_order_data
    loss = zero_one_loss(probs, targets)
    validate_metric(loss)

    probs, _ = sample_first_order_data
    loss = zero_one_loss(probs, None)
    validate_uncertainty(loss)


def test_spherical_score(
    sample_zero_order_data: tuple[np.array, np.array], sample_first_order_data: tuple[np.array, np.array]
) -> None:
    probs, targets = sample_zero_order_data
    loss = spherical_score(probs, targets)
    validate_metric(loss)

    probs, _ = sample_first_order_data
    loss = spherical_score(probs, None)
    validate_uncertainty(loss)
