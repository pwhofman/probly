"""Tests for the classification module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.utils import validate_uncertainty

if TYPE_CHECKING:
    from collections.abc import Callable

from probly.metrics import brier_score, log_loss, zero_one_loss
from probly.quantification.classification import (
    aleatoric_uncertainty_distance,
    conditional_entropy,
    epistemic_uncertainty_distance,
    evidential_uncertainty,
    expected_conditional_variance,
    expected_divergence,
    expected_entropy,
    expected_loss,
    generalised_hartley,
    lower_entropy,
    lower_entropy_convex_hull,
    mutual_information,
    total_entropy,
    total_uncertainty_distance,
    total_variance,
    upper_entropy,
    upper_entropy_convex_hull,
    variance_conditional_expectation,
)


@pytest.fixture
def sample_second_order_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    probs2d = rng.dirichlet(np.ones(2), (10, 5))
    probs3d = rng.dirichlet(np.ones(3), (10, 5))
    return probs2d, probs3d


@pytest.mark.parametrize(
    "uncertainty_fn",
    [
        total_entropy,
        conditional_entropy,
        mutual_information,
        total_variance,
        expected_conditional_variance,
        variance_conditional_expectation,
        total_uncertainty_distance,
        aleatoric_uncertainty_distance,
        epistemic_uncertainty_distance,
        upper_entropy,
        lower_entropy,
        upper_entropy_convex_hull,
        lower_entropy_convex_hull,
        generalised_hartley,
    ],
)
def test_uncertainty_function(
    uncertainty_fn: Callable[[np.ndarray], np.ndarray], sample_second_order_data: tuple[np.ndarray, np.ndarray]
) -> None:
    probs2d, probs3d = sample_second_order_data
    uncertainty = uncertainty_fn(probs2d)
    validate_uncertainty(uncertainty)

    uncertainty = uncertainty_fn(probs3d)
    validate_uncertainty(uncertainty)


@pytest.mark.parametrize("uncertainty_fn", [expected_loss, expected_entropy, expected_divergence])
def test_loss_uncertainty_function(
    uncertainty_fn: Callable[[np.ndarray, Callable[[np.ndarray, np.ndarray | None], np.ndarray]], np.ndarray],
    sample_second_order_data: tuple[np.ndarray, np.ndarray],
) -> None:
    probs2d, probs3d = sample_second_order_data
    for loss_fn in [log_loss, brier_score, zero_one_loss]:
        uncertainty = uncertainty_fn(probs2d, loss_fn)
        validate_uncertainty(uncertainty)

        uncertainty = uncertainty_fn(probs3d, loss_fn)
        validate_uncertainty(uncertainty)


def test_evidential_uncertainty() -> None:
    rng = np.random.default_rng()
    evidence = rng.uniform(0, 100, (10, 3))
    uncertainty = evidential_uncertainty(evidence)
    validate_uncertainty(uncertainty)
