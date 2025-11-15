""" Test for the bayesian models. """

from __future__ import annotations
import pytest

from probly.predictor import Predictor
from probly.transformation import bayesian

from torch import nn



class TestNetworkArchitectures:

# Tests if a linear model can be replaced by bayesian() using dummy parameters
    def test_replace_linear_model(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        model = bayesian(torch_model_small_2d_2d, True, 0.3, 1.5, 0.5)

        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))

# Tests if a convolutional model can be replaced by bayesian() using dummy parameters
    def test_replace_conv2d_model(self, torch_conv_linear_model: nn.Sequential) -> None:
        model = bayesian(torch_conv_linear_model, True, 0.3, 1.5, 0.5)

        assert model is not None
        assert isinstance(model, type(torch_conv_linear_model))
