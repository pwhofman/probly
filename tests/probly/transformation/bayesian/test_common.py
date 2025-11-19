"""Test for the bayesian models."""

from __future__ import annotations

import pytest

from probly.transformation import bayesian
from probly.predictor import Predictor





class TestNetworkArchitectures:

# Tests if a linear model can be replaced by bayesian() using dummy parameters
    def test_replace_linear_model(self, dummy_predictor: Predictor) -> None:
        # Transform the dummy_predictor with dummy parameters
        model = bayesian(dummy_predictor, True, 0.3, 1.5, 0.5)

        # Check if there is a new model and if bayesian() returned the correct type, Predictor
        assert model is not None
        assert isinstance(model, Predictor)


