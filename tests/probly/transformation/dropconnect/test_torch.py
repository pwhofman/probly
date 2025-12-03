"""Tests for torch dropconnect models."""

from __future__ import annotations

import pytest

from probly.transformation import dropconnect
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn

from probly.layers.torch import DropConnectLinear


class TestNetworkArchitectures:
    """Test class for different network architectures with DropConnect."""

    def test_linear_network_with_first_linear(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if a model incorporates DropConnect layers correctly when replacing linear layers.

        This function verifies that:
        - Linear layers are replaced by DropConnectLinear layers (except first layer)
        - The structure of the model remains unchanged except for the layer replacements
        - Only the specified probability parameter is applied in dropconnect modifications.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if
            the DropConnect layer is not inserted correctly.
        """
        p = 0.5
        model = dropconnect(torch_model_small_2d_2d, p)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)
        # count number of nn.DropConnectLinear layers in original model
        count_dropconnect_original = count_layers(torch_model_small_2d_2d, DropConnectLinear)

        # count number of DropConnectLinear layers in modified model
        count_dropconnect_modified = count_layers(model, DropConnectLinear)
        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)

        # check that linear layers are replaced by DropConnectLinear layers
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_dropconnect_modified == count_linear_original - 1
        assert count_linear_modified == 1
        assert count_dropconnect_original == 0
        assert count_sequential_original == count_sequential_modified

    def test_convolutional_network(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests the convolutional neural network modification with DropConnect layers.

        This function evaluates whether the given convolutional neural network model
        has been correctly modified to replace linear layers with DropConnectLinear layers
        without altering other components.

        Parameters:
            torch_conv_linear_model: The original convolutional neural network model to be tested.

        Raises:
            AssertionError: If the modified model deviates in structure other than
            the replacement of linear layers.
        """
        p = 0.5
        model = dropconnect(torch_conv_linear_model, p)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
        # count number of nn.Conv2d layers in original model
        count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)
        # count number of nn.DropConnectLinear layers in original model
        count_dropconnect_original = count_layers(torch_conv_linear_model, DropConnectLinear)

        # count number of DropConnectLinear layers in modified model
        count_dropconnect_modified = count_layers(model, DropConnectLinear)
        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)
        # count number of nn.Conv2d layers in modified model
        count_conv_modified = count_layers(model, nn.Conv2d)

        # check that the model is not modified except for the dropout layer
        assert model is not None
        assert isinstance(model, type(torch_conv_linear_model))
        assert count_dropconnect_original == 0
        assert count_dropconnect_modified == count_linear_original  # No skipping in conv models
        assert count_linear_modified == 0  # All linear layers should be replaced
        assert count_sequential_original == count_sequential_modified
        assert count_conv_original == count_conv_modified

        # Additional verification: the linear layer should be replaced with DropConnectLinear
        assert isinstance(model[3], DropConnectLinear), "Linear layer should be replaced with DropConnectLinear"

    def test_custom_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the custom model modification with DropConnect layers."""
        p = 0.5
        model = dropconnect(torch_custom_model, p)

        # check if model type is correct
        assert isinstance(model, type(torch_custom_model))
        assert not isinstance(model, nn.Sequential)

    def test_first_layer_skipping(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Test that the first linear layer is not replaced due to is_first_layer condition."""
        p = 0.5
        model = dropconnect(torch_model_small_2d_2d, p)

        # First layer should stay unchanged
        assert isinstance(model[0], nn.Linear), "First layer should remain nn.Linear"
        assert not isinstance(model[0], DropConnectLinear), "First layer should not be DropConnectLinear"

        # Linear layers should be replaced
        for i in range(1, len(model)):
            if isinstance(torch_model_small_2d_2d[i], nn.Linear):
                assert isinstance(model[i], DropConnectLinear), f"Layer {i} should be DropConnectLinear"

    def test_dropconnect_duplicates_behavior(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests that reapplying in dropconnect doesn't create duplicates."""
        first_p = 0.3
        model_after_first = dropconnect(torch_model_small_2d_2d, first_p)

        second_p = 0.5
        model_after_second = dropconnect(model_after_first, second_p)

        count_dropconnect_first = count_layers(model_after_first, DropConnectLinear)
        count_linear_first = count_layers(model_after_first, nn.Linear)

        count_dropconnect_second = count_layers(model_after_second, DropConnectLinear)
        count_linear_second = count_layers(model_after_second, nn.Linear)

        # Layers are replaced, reapplying should not change layer counts
        assert count_dropconnect_first == count_dropconnect_second
        assert count_linear_first == count_linear_second


class TestPValues:
    """Test class for p-value tests with DropConnect."""

    def test_linear_network_p_value(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests the DropConnect layer's p-value in a given neural network model.

        This function verifies that DropConnect layers inside the provided neural network
        model have the expected p-value after applying the dropconnect transformation.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested for integration

        Raises:
            AssertionError: If the p-value in a DropConnect layer does not match the expected value.
        """
        p = 0.5
        model = dropconnect(torch_model_small_2d_2d, p)

        # check p value in DropConnect layers (skip first layer)
        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert m.p == p

    def test_conv_network_p_value(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests whether DropConnect layers in the convolutional model have the correct probability value.

        Arguments:
            torch_conv_linear_model: A sequential model containing convolutional and linear layers.

        Raises:
            AssertionError: If the probability value in any DropCosnnect layer does not match the expected value.
        """
        p = 0.2
        model = dropconnect(torch_conv_linear_model, p)

        # check p value in DropConnect layers (skip first layer)
        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert m.p == p
