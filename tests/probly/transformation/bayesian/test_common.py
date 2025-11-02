"""test for bayesian models."""

from __future__ import annotations

from probly.predictor import Predictor
from probly.transformation import bayesian


def test_bayesian_noo_param(dummy_predictor: Predictor) -> None:
    """Test that bayesian can be called without prior or likelihood."""
    result = bayesian(dummy_predictor)

    assert result is not None
