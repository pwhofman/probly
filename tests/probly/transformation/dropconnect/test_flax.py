"""Tests for Flax DropConnect transformation."""

from __future__ import annotations

import pytest

from probly.layers.flax import DropConnectDense
from probly.transformation.dropconnect import dropconnect
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")

from flax import nnx  # noqa: E402


class TestNetworkArchitectures:
    """Structure tests for different network architectures with DropConnect."""

    def test_linear_network_starts_with_linear(
        self, flax_model_small_2d_2d: nnx.Sequential
    ) -> None:
        """
        If the network's first layer is Linear, DropConnect should *skip the first layer*
        and replace all subsequent Linear layers with DropConnectLinear.
        """
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # original counts
        linear_orig = count_layers(flax_model_small_2d_2d, nnx.Linear)
        seq_orig = count_layers(flax_model_small_2d_2d, nnx.Sequential)

        # modified counts
        linear_mod = count_layers(model, nnx.Linear)
        dc_mod = count_layers(model, DropConnectDense)
        seq_mod = count_layers(model, nnx.Sequential)

        # structure unchanged except Linear→DropConnectLinear replacements
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        # first Linear remains, rest replaced
        assert dc_mod == max(0, linear_orig - 1)
        assert linear_mod == min(1, linear_orig)
        # Sequential container count unchanged
        assert seq_orig == seq_mod

    def test_conv_then_linear_network(
        self, flax_conv_linear_model: nnx.Sequential
    ) -> None:
        """
        If the first layer is NOT Linear (e.g., Conv2d), *all* Linear layers
        should be replaced by DropConnectLinear.
        """
        p = 0.5
        model = dropconnect(flax_conv_linear_model, p)

        # original counts
        linear_orig = count_layers(flax_conv_linear_model, nnx.Linear)
        conv_orig = count_layers(flax_conv_linear_model, nnx.Conv)
        seq_orig = count_layers(flax_conv_linear_model, nnx.Sequential)

        # modified counts
        linear_mod = count_layers(model, nnx.Linear)
        dc_mod = count_layers(model, DropConnectDense)
        conv_mod = count_layers(model, nnx.Conv)
        seq_mod = count_layers(model, nnx.Sequential)

        # all linears replaced; convs and containers unchanged
        assert isinstance(model, type(flax_conv_linear_model))
        assert dc_mod == linear_orig
        assert linear_mod == 0
        assert conv_mod == conv_orig
        assert seq_mod == seq_mod

    def test_custom_network_keeps_type(self, flax_custom_model: nnx.Module) -> None:
        """Tests that transformation preserves the top-level model type.
        
        This function verifies that after applying DropConnect transformation,
        the custom model maintains its original type and is not wrapped in
        a Sequential container.
        
        Parameters:
            flax_custom_model: A custom Flax model (not Sequential).
            
        Raises:
            AssertionError: If the model type is changed after transformation.
        """
        p = 0.5
        model = dropconnect(flax_custom_model, p)
        
        # Check that after applying DropConnect, the top-level model type is unchanged
        assert model is not None
        assert isinstance(model, type(flax_custom_model))
        assert not isinstance(model, nnx.Sequential)


class TestPValues:
    """Test class for p-value tests."""

    def test_p_value_in_linear_first_model(
        self, flax_model_small_2d_2d: nnx.Sequential
    ) -> None:
        """Tests the DropConnect layer's p-value in a linear neural network model.
        
        This function verifies that DropConnectDense layers inside the provided
        neural network model have the expected p-value after applying the
        dropconnect transformation. The p-value represents the probability of
        dropping a weight connection during training.
        
        Parameters:
            flax_model_small_2d_2d: The Flax model to be tested.
            
        Raises:
            AssertionError: If the p-value in a DropConnectDense layer does not
                match the expected value.
        """
        p = 0.3
        model = dropconnect(flax_model_small_2d_2d, p)
        
        # Check p value in DropConnectDense layers
        for _, m in model.iter_modules():
            if isinstance(m, DropConnectDense):
                # DropConnectDense exposes .p
                assert m.p == p

    def test_p_value_in_conv_model(
        self, flax_conv_linear_model: nnx.Sequential
    ) -> None:
        """Tests the DropConnect layer's p-value in a convolutional model.
        
        This function verifies that DropConnectDense layers in a convolutional
        neural network have the correct probability value after applying the
        dropconnect transformation.
        
        Parameters:
            flax_conv_linear_model: A sequential model containing convolutional
                and linear layers.
                
        Raises:
            AssertionError: If the probability value in any DropConnectDense layer
                does not match the expected value.
        """
        p = 0.2
        model = dropconnect(flax_conv_linear_model, p)
        
        # Check p value in DropConnectDense layers
        for _, m in model.iter_modules():
            if isinstance(m, DropConnectDense):
                # DropConnectDense exposes .p
                assert m.p == p