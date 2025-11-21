from __future__ import annotations

import pytest
from torch import nn

from probly.transformation.bayesian import torch
from probly.layers.torch import BayesLinear, BayesConv2d
from tests.probly.torch_utils import count_layers
from probly.transformation import bayesian


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """This function verifies that every linear layer gets replaced by an bayesian layer."""
        """And that the structure of the model remains unchanged except for the added bayesian layers."""
        model = bayesian(torch_model_small_2d_2d, False, 0.3, 1.5, 0.5)

        # Count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        # Count number of BayesLinear layers in original model
        count_bayesian_original = count_layers(torch_model_small_2d_2d, BayesLinear)
        # Count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

        # Count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # Count number of BayesLinear layers in modified model
        count_bayesian_modified = count_layers(model, BayesLinear)
        # Count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)

        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_linear_original == count_bayesian_modified
        assert count_linear_modified == 0
        assert count_bayesian_original == 0
        assert count_sequential_original == count_sequential_modified




    def test_convolutional_network(self, torch_conv_linear_model: nn.Sequential) -> None:
        """
        This function verifies that:
        - Every convolutional layer gets replaced by an bayesian layer.
        - The structure of the model remains unchanged except for the added bayesian layers.
        """

        model = bayesian(torch_conv_linear_model, True, 0.3, 1.5, 0.5)

        # Count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        # Count number of nn.Dropout layers in original model
        count_bayesian_original = count_layers(torch_conv_linear_model, BayesConv2d)
        # Count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)

        # Count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # Count number of nn.Dropout layers in modified model
        count_bayesian_modified = count_layers(model, BayesConv2d)
        # Count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)



        assert model is not None
        assert isinstance(model, type(torch_conv_linear_model))
        assert count_linear_original == count_bayesian_modified
        assert count_linear_modified == 0
        assert count_bayesian_original == 0
        assert count_sequential_original == count_sequential_modified
