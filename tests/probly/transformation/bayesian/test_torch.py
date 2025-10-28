"""test for torch bayesian models."""

from __future__ import annotations

import pytest
from torch import nn

from probly.transformation import bayesian

torch = pytest.importorskip("torch")

def test_invalid_prior_likelihood(torch_model_small_2d_2d: nn.Module) -> None:
    """Tests the behavior of the bayesian function when provided with invalid prior or likelihood."""
    prior = [0.5, 0.6]  # Invalid prior (sums to more than 1)
    likelihood = [0.8, 0.2]
    with pytest.raises(ValueError, match="The prior must sum to 1."):
        bayesian(torch_model_small_2d_2d, prior=prior, likelihood=likelihood)

    prior = [0.5, 0.5]
    likelihood = [0.8, -0.2]  # Invalid likelihood (negative value)
    with pytest.raises(ValueError, match="The likelihood must be non-negative."):
        bayesian(torch_model_small_2d_2d, prior=prior, likelihood=likelihood)