"""Tests for the generic bayesian models transformation logic."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import bayesian


@pytest.mark.parametrize(
    (
        "use_base_weights",
        "posterior_std",
        "prior_mean",
        "prior_std",
    ),
    [
        (True, 0.5, 0.0, 1.0),
        (False, 0.01, 10.0, 5.0),
        (True, 0.05, -5.0, 0.1),
    ],
)
def test_bayesian_parameter_passing_and_type(
    dummy_predictor: Predictor,
    use_base_weights: bool,
    posterior_std: float,
    prior_mean: float,
    prior_std: float,
) -> None:
    """Tests Bayesian parameter acceptance and correct type return.

    This ensures the return type requirement is fulfilled and verifies all arguments
    can be passed without immediate generic failure.

    Parameters:
        dummy_predictor (Predictor): A generic Predictor fixture for testing.
        use_base_weights (bool): Whether to use base weights as prior mean.
        posterior_std (float): The initial posterior standard deviation.
        prior_mean (float): The prior mean.
        prior_std (float): The prior standard deviation.
    """
    # 1. Apply the bayesian transformation with custom parameters
    result = bayesian(
        dummy_predictor,
        use_base_weights=use_base_weights,
        posterior_std=posterior_std,
        prior_mean=prior_mean,
        prior_std=prior_std,
    )

    # 2. Assert that the return type is the same as the input type
    assert isinstance(result, type(dummy_predictor))

    # 3. Basic check that the result is not None
    assert result is not None


def test_bayesian_default_call(dummy_predictor: Predictor) -> None:
    """Tests default Bayesian call and correct return type.

    Parameters:
        dummy_predictor (Predictor): A generic Predictor fixture for testing.
    """
    # 1. Apply the bayesian transformation using default parameters
    result = bayesian(dummy_predictor)

    # 2. Assert that the return type is the same as the input type
    assert isinstance(result, type(dummy_predictor))

    # 3. Basic check that the result is not None
    assert result is not None
