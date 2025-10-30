"""Test for torch bayesian models."""

from __future__ import annotations

import pytest

from probly.layers.torch import BayesConv2d, BayesLinear
from probly.transformation import bayesian
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network_replacement(
        self,
        torch_model_small_2d_2d: nn.Sequential,
    ) -> None:
        """Tests if a model incorporates a bayesian layer correctly when a linear layer is present.

        This function verifies that:
        - A standard linear layer is replaced with a bayesian linear layer.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the modified model deviates in structure other than
            the replacement of bayesian layers or does not meet the expected constraints.
        """
        model = bayesian(torch_model_small_2d_2d)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        # count number of BayesLinear layers in original model
        count_bayesian_original = count_layers(torch_model_small_2d_2d, BayesLinear)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of BayesLinear layers in modified model
        count_bayesian_modified = count_layers(model, BayesLinear)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)

        # check that the model is not modified except for the bayesian layer
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_bayesian_modified == count_bayesian_original + count_linear_original
        assert count_linear_modified == 0
        assert count_sequential_original == count_sequential_modified

    def test_convolutional_network(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests the convolutional neural network modification with added bayesian layers.

        This function evaluates whether the given convolutional neural network model
        has been correctly modified to include bayesian layers without altering the
        number of other components such as linear, sequential, or convolutional layers.

        Parameters:
            torch_conv_linear_model: The original convolutional neural network model to be tested.

        Raises:
            AssertionError: If the modified model deviates in structure other than
            the addition of bayesian layers or does not meet the expected constraints.
        """
        model = bayesian(torch_conv_linear_model)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        # count number of BayesConv2d layers in original model
        count_bayesian_conv_original = count_layers(torch_conv_linear_model, BayesConv2d)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
        # count number of nn.Conv2d layers in original model
        count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)
        # count number of BayesLinear layers in original model
        count_bayesian_linear_original = count_layers(torch_conv_linear_model, BayesLinear)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of BayesConv2d layers in modified model
        count_bayesian_conv_modified = count_layers(model, BayesConv2d)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)
        # count number of nn.Conv2d layers in modified model
        count_conv_modified = count_layers(model, nn.Conv2d)
        # count number of BayesLinear layers in modified model
        count_bayesian_linear_modified = count_layers(model, BayesLinear)

        # check that the model is not modified except for the bayesian layer
        assert model is not None
        assert isinstance(model, type(torch_conv_linear_model))
        assert count_linear_modified == 0
        assert count_conv_modified == 0
        assert count_bayesian_conv_modified == count_bayesian_conv_original + count_conv_original
        assert count_bayesian_linear_modified == count_bayesian_linear_original + count_linear_original
        assert count_sequential_original == count_sequential_modified

    def test_custom_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the custom model modification with added bayesian layers."""
        model = bayesian(torch_custom_model)

        # check if model type is correct
        assert isinstance(model, type(torch_custom_model))
        assert not isinstance(model, nn.Sequential)

    @pytest.mark.skip(reason="Not yet implemented in probly")
    def test_bayesian_model(self, torch_bayesian_model: nn.Module) -> None:
        """Tests the bayesian model modification if bayesian already exists."""
        model = bayesian(torch_bayesian_model)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_bayesian_model, nn.Linear)
        # count number of BayesianLinear layers in original model
        count_bayesian_linear_original = count_layers(torch_bayesian_model, BayesLinear)
        # count number of BayesianLinear layers in modified model
        count_bayesian__linear_modified = count_layers(model, BayesLinear)

        # count number of nn.Conv2d layers in original model
        count_conv_original = count_layers(torch_bayesian_model, nn.Conv2d)
        # count number of BayesianConv2d layers in original model
        count_bayesian_conv_original = count_layers(torch_bayesian_model, BayesConv2d)
        # count number of BayesianConv2d layers in modified model
        count_bayesian_conv_modified = count_layers(model, BayesConv2d)

        # check that no nn.Linear and nn.Conv2d layers are present and bayesian layers are unchanged
        assert count_bayesian__linear_modified == count_bayesian_linear_original
        assert count_linear_original == 0
        assert count_bayesian_conv_modified == count_bayesian_conv_original
        assert count_conv_original == 0

        # check values in bayesian layers
        for m in model.modules():
            if isinstance(m, BayesLinear):
                msg = "Possible value checks for BayesianLinear"
                raise NotImplementedError(msg)
            if isinstance(m, BayesConv2d):
                msg = "Possible value checks for BayesianConv2d"
                raise NotImplementedError(msg)
