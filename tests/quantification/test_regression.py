"""Tests for the regression module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


from probly.quantification.regression import (
    conditional_differential_entropy,
    expected_conditional_variance,
    mutual_information,
    total_differential_entropy,
    total_variance,
    variance_conditional_expectation,
)


@pytest.fixture
def sample_second_order_params() -> np.ndarray:
    rng = np.random.default_rng()
    mu = rng.uniform(0, 1000, (5, 10))
    sigma2 = rng.uniform(1e-20, 100, (5, 10))
    params = np.stack((mu, sigma2), axis=2)
    return params


def validate_uncertainty(uncertainty: np.array) -> None:
    assert isinstance(uncertainty, np.ndarray)
    assert not np.isnan(uncertainty).any()
    assert not np.isinf(uncertainty).any()
    assert (uncertainty >= 0).all()


@pytest.mark.parametrize(
    "uncertainty_fn",
    [
        total_variance,
        expected_conditional_variance,
        variance_conditional_expectation,
        total_differential_entropy,
        conditional_differential_entropy,
        mutual_information,
    ],
)
def test_uncertainty_function(
    uncertainty_fn: Callable[[np.ndarray], np.ndarray], sample_second_order_params: np.ndarray
) -> None:
    uncertainty = uncertainty_fn(sample_second_order_params)
    validate_uncertainty(uncertainty)
