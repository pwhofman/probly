"""Test for torch dropconnect models."""

from __future__ import annotations

import pytest

from probly.layers.torch import DropConnectLinear
from probly.transformation import dropconnect
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network_with_first_linear(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if a model replaces linear layers correctly with DropConnectLinear layers.

        This function verifies that:
        - Linear layers are replaced by DropConnectLinear layers, except for the first layer.
        - The structure of the model remains unchanged except for the replaced layers.
        - Only the specified probability parameter is applied in dropconnect modifications.

        It performs counts and asserts to ensure the modified model adheres to expectations.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError If the structure of the model differs in an unexpected manner or if the layers are not
            replaced correctly.
        """
        p = 0.5
        model = dropconnect(torch_model_small_2d_2d, p)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        # count number of DropConnectLinear layers in original model
        count_dropconnect_original = count_layers(torch_model_small_2d_2d, DropConnectLinear)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of DropConnectLinear layers in modified model
        count_dropconnect_modified = count_layers(model, DropConnectLinear)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)

        # check that the model is modified by replacing Linear layers with DropConnectLinear
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert (count_linear_original - 1) == count_dropconnect_modified
        assert count_linear_original == count_linear_modified + count_dropconnect_modified
        assert count_dropconnect_original == 0
        assert count_sequential_original == count_sequential_modified

    def test_convolutional_network(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests the convolutional neural network modification with DropConnectLinear layers.

        This function evaluates whether the given convolutional neural network model
        has been correctly modified to replace linear layers with DropConnectLinear layers
        without altering the number of other components such as sequential or convolutional layers.

        Parameters:
            torch_conv_linear_model: The original convolutional neural network model to be tested.

        Raises:
            AssertionError: If the modified model deviates in structure other than
            the replacement of linear layers or does not meet the expected constraints.
        """
        p = 0.5
        model = dropconnect(torch_conv_linear_model, p)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        # count number of DropConnectLinear layers in original model
        count_dropconnect_original = count_layers(torch_conv_linear_model, DropConnectLinear)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
        # count number of nn.Conv2d layers in original model
        count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of DropConnectLinear layers in modified model
        count_dropconnect_modified = count_layers(model, DropConnectLinear)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)
        # count number of nn.Conv2d layers in modified model
        count_conv_modified = count_layers(model, nn.Conv2d)

        # check that the model is modified by replacing Linear layers with DropConnectLinear
        assert model is not None
        assert isinstance(model, type(torch_conv_linear_model))
        assert count_linear_original == count_dropconnect_modified
        assert count_linear_original == count_linear_modified + count_dropconnect_modified
        assert count_dropconnect_original == 0
        assert count_sequential_original == count_sequential_modified
        assert count_conv_original == count_conv_modified

    def test_custom_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the custom model modification with DropConnectLinear layers."""
        p = 0.5
        model = dropconnect(torch_custom_model, p)

        # check if model type is correct
        assert isinstance(model, type(torch_custom_model))
        assert not isinstance(model, nn.Sequential)

    @pytest.mark.skip(reason="Not yet implemented in probly")
    def test_dropconnect_model(self, torch_dropconnect_model: nn.Sequential) -> None:
        """Tests the dropconnect model modification if dropconnect already exists."""
        p = 0.2
        model = dropconnect(torch_dropconnect_model, p)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_dropconnect_model, nn.Linear)
        # count number of DropConnectLinear layers in original model
        count_dropconnect_original = count_layers(torch_dropconnect_model, DropConnectLinear)
        # count number of DropConnectLinear layers in modified model
        count_dropconnect_modified = count_layers(model, DropConnectLinear)

        # check that model has no duplicate dropconnect layers
        assert count_dropconnect_original == 1
        assert count_linear_original == 2
        assert (count_linear_original - 1) == count_dropconnect_modified

        # check p value in dropconnect layer
        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert m.p == p


class TestPValues:
    """Test class for p-value tests."""

    def test_linear_network_p_value(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests the DropConnectLinear layer's p-value in a given neural network model.

        This function verifies that a DropConnectLinear layer inside the provided neural network
        model has the expected p-value after applying the dropconnect transformation. The
        p-value represents the probability of a connection being dropped during training.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested for integration

        Raises:
            AssertionError: If the p-value in a DropConnectLinear layer does not match the expected value.
        """
        p = 0.5
        model = dropconnect(torch_model_small_2d_2d, p)

        # check p value in dropconnect layer
        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert m.p == p

    def test_conv_network_p_value(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests whether the DropConnectLinear layer in the convolutional model has the correct probability value.

        Arguments:
            torch_conv_linear_model: A sequential model containing convolutional and linear layers.

        Raises:
            AssertionError: If the probability value in any DropConnectLinear layer does not match the expected value.
        """
        p = 0.2
        model = dropconnect(torch_conv_linear_model, p)

        # check p value in dropconnect layer
        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert m.p == p
