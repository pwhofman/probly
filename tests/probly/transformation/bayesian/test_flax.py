"""test for flax bayesian models."""

from __future__ import annotations

import pytest

from probly.transformation import bayesian

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


class TestBayesianTransformation:
    """Test class for Bayesian transformation."""

    def test_invalid_prior_likelihood(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests the behavior of the bayesian function when provided with invalid prior or likelihood.

        This function validates that the bayesian function raises a ValueError when
        the prior or likelihood parameters are not valid probability distributions.

        Raises:
            ValueError: If the prior or likelihood are not valid probability distributions.
        """
        prior = [0.5, 0.6]  # Invalid prior (sums to more than 1)
        likelihood = [0.8, 0.2]
        with pytest.raises(ValueError, match="The prior must sum to 1."):
            bayesian(flax_model_small_2d_2d, prior=prior, likelihood=likelihood)

        prior = [0.5, 0.5]
        likelihood = [0.8, -0.2]  # Invalid likelihood (negative value)
        with pytest.raises(ValueError, match="The likelihood must be non-negative."):
            bayesian(flax_model_small_2d_2d, prior=prior, likelihood=likelihood)