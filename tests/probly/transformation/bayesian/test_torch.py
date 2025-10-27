"""Test for torch bayesian models."""

from __future__ import annotations

import pytest

from probly.transformation import bayesian
from tests.probly.torch_utils import count_layers
from probly.layers.torch import BayesLinear, BayesConv2d

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network_with_first_linear(
        self, torch_model_small_2d_2d: nn.Sequential
    ) -> None:
        """Tests if a model incorporates a bayesian layer correctly when a linear layer succeeds it.

        This function verifies that:
        - A bayesian layer is added before each linear layer in the model, except for the last linear layer.
        - The structure of the model remains unchanged except for the added bayesian layers.
        - Only the specified probability parameter is applied in bayesian modifications.

        It performs counts and asserts to ensure the modified model adheres to expectations.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the bayesian layer is not
            inserted correctly after linear layers.
        """
        p = 0.5
        model = bayesian(torch_model_small_2d_2d, p)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        # count number of nn.bayesian layers in original model
        count_bayesian_original = count_layers(torch_model_small_2d_2d, BayesLinear)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of nn.bayesian layers in modified model
        count_bayesian_modified = count_layers(model, BayesLinear)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)

        # check that the model is not modified except for the bayesian layer
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert (count_linear_original - 1) == count_bayesian_modified
        assert count_linear_modified == count_linear_original
        assert count_bayesian_original == 0
        assert count_sequential_original == count_sequential_modified

    def test_convolutional_network(
        self, torch_conv_linear_model: nn.Sequential
    ) -> None:
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
        p = 0.5
        model = bayesian(torch_conv_linear_model, p)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        # count number of nn.bayesian layers in original model
        count_bayesian_original = count_layers(torch_conv_linear_model, BayesConv2d)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
        # count number of nn.Conv2d layers in original model
        count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of nn.bayesian layers in modified model
        count_bayesian_modified = count_layers(model, BayesConv2d)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)
        # count number of nn.Conv2d layers in modified model
        count_conv_modified = count_layers(model, nn.Conv2d)

        # check that the model is not modified except for the bayesian layer
        assert model is not None
        assert isinstance(model, type(torch_conv_linear_model))
        assert count_linear_original == count_bayesian_modified
        assert count_linear_original == count_linear_modified
        assert count_bayesian_original == 0
        assert count_sequential_original == count_sequential_modified
        assert count_conv_original == count_conv_modified

    def test_custom_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the custom model modification with added bayesian layers."""
        p = 0.5
        model = bayesian(torch_custom_model, p)

        # check if model type is correct
        assert isinstance(model, type(torch_custom_model))
        assert not isinstance(model, nn.Sequential)

    @pytest.mark.skip(reason="Not yet implemented in probly")
    def test_bayesian_model(self, torch_bayesian_model: nn.Sequential) -> None:
        """Tests the bayesian model modification if bayesian already exists."""
        p = 0.2
        model = bayesian(torch_bayesian_model, p)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_bayesian_model, nn.Linear)
        # count number of nn.bayesian layers in original model
        count_bayesian_original = count_layers(torch_bayesian_model, BayesLinear)
        # count number of nn.bayesian layers in modified model
        count_bayesian_modified = count_layers(model, BayesLinear)

        # check that model has no duplicate bayesian layers
        assert count_bayesian_original == 1
        assert count_linear_original == 2
        assert (count_linear_original - 1) == count_bayesian_modified

        # check p value in bayesian layer
        for m in model.modules():
            if isinstance(m, nn.bayesian):
                assert m.p == p


class TestPValues:
    """Test class for p-value tests."""

    def test_linear_network_p_value(
        self, torch_model_small_2d_2d: nn.Sequential
    ) -> None:
        """Tests the bayesian layer's p-value in a given neural network model.

        This function verifies that a bayesian layer inside the provided neural network
        model has the expected p-value after applying the bayesian transformation. The
        p-value represents the probability of an element being zeroed during training.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested for integration

        Raises:
            AssertionError: If the p-value in a bayesian layer does not match the expected value.
        """
        p = 0.5
        model = bayesian(torch_model_small_2d_2d, p)

        # check p value in bayesian layer
        for m in model.modules():
            if isinstance(m, BayesLinear):
                assert m.p == p

    def test_conv_network_p_value(self, torch_conv_linear_model: nn.Sequential) -> None:
        """This function tests whether the bayesian layer in the convolutional model has the correct probability value.

        Arguments:
            torch_conv_linear_model: A sequential model containing convolutional and linear layers.

        Raises:
            AssertionError: If the probability value in any bayesian layer does not match the expected value.
        """
        p = 0.2
        model = bayesian(torch_conv_linear_model, p)

        # check p value in bayesian layer
        for m in model.modules():
            if isinstance(m, BayesConv2d):
                assert m.p == p
