"""Test for torch dropconnect models."""

from __future__ import annotations

import pytest

from probly.transformation import dropconnect
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestNetworkArchitectures:
    """Test class fpr different network architectures."""

    def test_linear_network_replacement(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Testst if Linear layers are correctly replaced with DropConnectLinear layers.

        This function verifies that:
        - Linear layers in the model are replaced with DropConnectLinear layers.
        - The structure of the model remains unchanged.
        - The probability parameter is correctly applied.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner.
        """
        p = 0.5
        model = dropconnect(torch_model_small_2d_2d, p)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        # count number of DropConnectLinear layers in modified model
        count_dropconnect_linear_modified = count_layers(model, nn.DropConnectLinear)

        # check that Linear layers are replaced with DropConnectLinear
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_linear_original == count_dropconnect_linear_modified

    def test_convolutional_network(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests DropConnect in convolutional neural networks.

        This function evaluates whether only Linear layers are replaced with DropConnectLinear
        while convolutional layers remain unchanged.

        Parameters:
            torch_conv_linear_model: The torch convolutional model to be tested.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner.
        """
        p = 0.5
        model = dropconnect(torch_conv_linear_model, p)

        # count layers in original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)
        count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)

        # count layers in modified model
        count_dropconnect_linear_modified = count_layers(model, nn.DropConnectLinear)
        count_conv_modified = count_layers(model, nn.Conv2d)
        count_sequential_modified = count_layers(model, nn.Sequential)

        # check that only Linear layers are replaced with DropConnectLinear
        assert model is not None
        assert isinstance(model, type(torch_conv_linear_model))
        assert count_linear_original == count_dropconnect_linear_modified
        assert count_conv_original == count_conv_modified
        assert count_sequential_original == count_sequential_modified

    def test_custom_network_dropconnect(self, torch_custom_model: nn.Module) -> None:
        """Tests DropConnect in a custom neural network model."""
        p = 0.5
        model = dropconnect(torch_custom_model, p)

        # check if model type is preserved
        assert isinstance(model, type(torch_custom_model))
        assert not isinstance(model, nn.Sequential)


class TestPValues:
    """Test class for p-value tests."""

    def test_linear_network_p_value(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests the DropConnect layers p-value is in a neural network model.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested.

        Raises:
            AssertionError: If the p-value in DropConnect layers does not match the specified value.
        """
        p = 0.5
        model = dropconnect(torch_model_small_2d_2d, p)

        # check p-value in DropConnect layers
        for m in model.modules():
            if hasattr(m, "__class__") and m.__class__.__name__ == "DropConnectLinear":
                assert hasattr(m, "p")
                assert m.p == p

    def test_convolutional_network_p_value(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests p-values in DropConnect layers of convolutional models.

        Arguments:
            torch_conv_linear_model: A sequential model containing conv and linear layers.

        Raises:
            AssertionError: If the p-value in DropConnect layers does not match the specified value.
        """
        p = 0.2
        model = dropconnect(torch_conv_linear_model, p)

        # check p-value in DropConnect layers
        for m in model.modules():
            if hasattr(m, "__class__") and m.__class__.__name__ == "DropConnectLinear":
                assert hasattr(m, "p")
                assert m.p == p
