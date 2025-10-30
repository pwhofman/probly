"""Test for dropout models."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import bayesian


def test_invalid_prior_std_value(dummy_predictor: Predictor) -> None:
    """Tests the behavior of the bayesian function when provided with an invalid prior standard deviation value.

    This function validates that the bayesian function raises a ValueError when
    the prior standard deviation parameter is not positive.

    Raises:
        ValueError: If the prior standard deviation is not positive or equal to zero.
    """
    prior_std = -1.0
    msg = f"The prior standard deviation must be positive and non-zero, but got {prior_std} instead."
    with pytest.raises(ValueError, match=msg):
        bayesian(dummy_predictor, prior_std=prior_std)


def test_invalid_posterior_std_value(dummy_predictor: Predictor) -> None:
    """Tests the behavior of the bayesian function when provided with an invalid posterior standard deviation value.

    This function validates that the bayesian function raises a ValueError when
    the posterior standard deviation parameter is not positive.

    Raises:
        ValueError: If the posterior standard deviation is not positive or equal to zero.
    """
    posterior_std = -1.0
    msg = f"The posterior standard deviation must be positive and non-zero, but got {posterior_std} instead."
    with pytest.raises(ValueError, match=msg):
        bayesian(dummy_predictor, posterior_std=posterior_std)
