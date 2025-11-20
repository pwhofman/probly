"""Test for flax dropconnect models."""

from __future__ import annotations

import pytest

pytest.importorskip("flax")
from flax import nnx

from probly.layers.my_flax import DropConnectLinear
from probly.transformation import dropconnect
from tests.probly.flax_utils import count_layers


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network_replacement(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if Linear layers are correctly replaced with DropConnectLinear layers.

        This function verifies that:
        - Linear layers in the model are replaced with DropConnectLinear layers.
        - The structure of the model remains unchanged.
        - The probability parameter is correctly applied.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner.
        """
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        # count number of DropConnectLinear layers in modified model
        count_dropconnect_linear_modified = count_layers(model, DropConnectLinear)

        # check that Linear layers are replaced with DropConnectLinear
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert count_linear_original == count_dropconnect_linear_modified + 1

    def test_convolutional_network(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests DropConnect in convolutional neural networks.

        This function evaluates whether only Linear layers are replaced with DropConnectLinear
        while convolutional layers remain unchanged.

        Parameters:
            flax_conv_linear_model: The flax convolutional model to be tested.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner.
        """
        p = 0.5
        model = dropconnect(flax_conv_linear_model, p)

        # count layers in original model
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv)
        count_sequential_original = count_layers(flax_conv_linear_model, nnx.Sequential)

        # count layers in modified model
        count_dropconnect_linear_modified = count_layers(model, DropConnectLinear)
        count_conv_modified = count_layers(model, nnx.Conv)
        count_sequential_modified = count_layers(model, nnx.Sequential)

        # check that only Linear layers are replaced with DropConnectLinear
        assert model is not None
        assert isinstance(model, type(flax_conv_linear_model))
        assert count_linear_original == count_dropconnect_linear_modified
        assert count_conv_original == count_conv_modified
        assert count_sequential_original == count_sequential_modified

    def test_custom_network_dropconnect(self, flax_custom_model: nnx.Module) -> None:
        """Tests DropConnect in a custom flax neural network."""
        p = 0.5
        model = dropconnect(flax_custom_model, p)

        # check if model type is preserved
        assert isinstance(model, type(flax_custom_model))
        assert not isinstance(model, nnx.Sequential)


class TestPValues:
    """Test class for p-value tests."""

    def test_linear_network_p_value(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests the DropConnect layers p-value is in a Flax network model.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested.

        Raises:
            AssertionError: If the p-value in DropConnect layers does not match the specified value.
        """
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # check p-value in DropConnectLinear layers
        for m in getattr(model, "layers", []):
            if hasattr(m, "__class__") and m.__class__.__name__ == "DropConnectLinear":
                assert hasattr(m, "p")
                assert m.p == p

    def test_convolutional_network_p_value(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests p-values in DropConnect layers of a Flax convolutional models.

        Parameters:
            flax_conv_linear_model: A sequential model containing conv and linear layers.

        Raises:
            AssertionError: If the p-value in DropConnect layers does not match the specified value.
        """
        p = 0.2
        model = dropconnect(flax_conv_linear_model, p)

        # check p-value in DropConnect layers
        for m in getattr(model, "layers", []):
            if hasattr(m, "__class__") and m.__class__.__name__ == "DropConnectLinear":
                assert hasattr(m, "p")
                assert m.p == p
