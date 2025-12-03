"""Test for flax dropconnect models."""

from __future__ import annotations

import pytest

flax = pytest.importorskip("flax", reason="Flax not installed, skipping DropConnect tests")
jax = pytest.importorskip("jax", reason="JAX not installed, skipping DropConnect tests")

from flax import nnx  # noqa: E402

from probly.transformation import dropconnect
from tests.probly.flax_utils import count_layers


class TestNetworkArchitectures:
    """Test class for different network architectures with DropConnect."""

    def test_linear_network_with_first_linear(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if a model incorporates DropConnect layers correctly when replacing linear layers."""
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # Count layers
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        count_dropconnect_modified = count_layers(model, nnx.DropConnectLinear)
        count_linear_modified = count_layers(model, nnx.Linear)

        # Check layer replacement
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert count_dropconnect_modified == count_linear_original - 1  # First Layer skipped
        assert count_linear_modified == 1  # Only first linear layer remains

    def test_convolutional_network(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests the convolutional neural network modification with DropConnect layers."""
        p = 0.5
        model = dropconnect(flax_conv_linear_model, p)

        # Count layers
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv2d)
        count_dropconnect_modified = count_layers(model, nnx.DropConnectLinear)
        count_linear_modified = count_layers(model, nnx.Linear)
        count_conv_modified = count_layers(model, nnx.Conv2d)

        # Check layer replacement
        assert model is not None
        assert isinstance(model, type(flax_conv_linear_model))
        assert count_dropconnect_modified == count_linear_original  # No skipping in conv models
        assert count_linear_modified == 0  # All linear layers replaced
        assert count_conv_original == count_conv_modified  # Conv layers unchanged

    def test_custom_network(self, flax_custom_model: nnx.Module) -> None:
        """Tests the custom model modification with DropConnect layers."""
        p = 0.5
        model = dropconnect(flax_custom_model, p)

        # Check model type
        assert isinstance(model, type(flax_custom_model))

    def test_first_layer_skipping(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Test that the first linear layer is not replaced."""
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # First layer should remain unchanged
        assert isinstance(model.layers[0], nnx.Linear), "First layer should remain nnx.Linear"
        assert not isinstance(model.layers[0], nnx.DropConnectLinear), "First layer should not be DropConnectLinear"

        # Subsequent linear layers should be replaced
        for i in range(1, len(model.layers)):
            if isinstance(flax_model_small_2d_2d.layers[i], nnx.Linear):
                assert isinstance(model.layers[i], nnx.DropConnectLinear), f"Layer {i} should be DropConnectLinear"

    def test_dropconnect_duplicates_behavior(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests that reapplying dropconnect doesn't create duplicates."""
        first_p = 0.3
        model_after_first = dropconnect(flax_model_small_2d_2d, first_p)

        second_p = 0.5
        model_after_second = dropconnect(model_after_first, second_p)

        # Layer counts should not change
        count_dropconnect_first = count_layers(model_after_first, nnx.DropConnectLinear)
        count_linear_first = count_layers(model_after_first, nnx.Linear)
        count_dropconnect_second = count_layers(model_after_second, nnx.DropConnectLinear)
        count_linear_second = count_layers(model_after_second, nnx.Linear)

        assert count_dropconnect_first == count_dropconnect_second
        assert count_linear_first == count_linear_second


class TestPValues:
    """Test class for p-value tests with DropConnect."""

    def test_linear_network_p_value(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests the DropConnect layer's p-value."""
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # Check p values in DropConnect layers
        for layer in model.layers:
            if isinstance(layer, nnx.DropConnectLinear):
                assert layer.p == p

    def test_conv_network_p_value(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests whether DropConnect layers have the correct probability value."""
        p = 0.2
        model = dropconnect(flax_conv_linear_model, p)

        # Check p values in DropConnect layers
        for layer in model.layers:
            if isinstance(layer, nnx.DropConnectLinear):
                assert layer.p == p
