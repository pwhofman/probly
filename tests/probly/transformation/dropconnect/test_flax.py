"""Test for flax dropconnect models."""

from __future__ import annotations

import pytest

from probly.transformation import dropconnect
from tests.probly.flax_utils import count_layers
from probly.layers.flax import DropConnectLinear, Conv2d
from typing import Any, cast

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network_with_first_linear(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if a model incorporates DropConnectLinear layers correctly when replacing linear layers."""
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        # count number of DropConnectLinear layers in original model
        count_dropconnect_original = count_layers(flax_model_small_2d_2d, DropConnectLinear)
        
        # count number of DropConnectLinear layers in modified model
        count_dropconnect_modified = count_layers(model, DropConnectLinear)
        # count number of nnx.Linear layers in modified model
        count_linear_modified = count_layers(model, nnx.Linear)

        # check that linear layers are replaced by DropConnectLinear layers 
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert count_dropconnect_modified == count_linear_original - 1  
        assert count_linear_modified == 1  
        assert count_dropconnect_original == 0

    def test_convolutional_network(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests the convolutional neural network modification with DropConnectLinear layers."""
        p = 0.5
        model = dropconnect(flax_conv_linear_model, p)
        
        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)
        # count number of Conv2d layers in original model 
        count_conv_original = count_layers(flax_conv_linear_model, Conv2d)
        # count number of DropConnectLinear layers in original model
        count_dropconnect_original = count_layers(flax_conv_linear_model, DropConnectLinear)

        # count number of DropConnectLinear layers in modified model
        count_dropconnect_modified = count_layers(model, DropConnectLinear)
        # count number of nnx.Linear layers in modified model
        count_linear_modified = count_layers(model, nnx.Linear)
        # count number of Conv2d layers in modified model
        count_conv_modified = count_layers(model, Conv2d)

        assert model is not None
        assert isinstance(model, type(flax_conv_linear_model))
        assert count_dropconnect_original == 0 
        assert count_dropconnect_modified == count_linear_original
        assert count_linear_modified == 0
        assert count_conv_original == count_conv_modified
            
        # Additional verification: the linear layer should be replaced with DropConnectLinear
        assert isinstance(model.layers[3], DropConnectLinear), "Linear layer should be replaced with DropConnectLinear"



class TestPValues:
    """Test class for p-value tests with DropConnectLinear."""

    def test_linear_network_p_value(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests DropConnectLinear layer's p-value in a given neural network model."""
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # check p value in DropConnectLinear layers
        for layer in model.layers:
            if isinstance(layer, DropConnectLinear):
                assert layer.p == p

    def test_conv_network_p_value(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests whether DropConnectLinear layers have the correct probability value."""
        p = 0.2
        model = dropconnect(flax_conv_linear_model, p)

        # check p value in DropConnectLinear layers 
        for layer in model.layers:
            if isinstance(layer, DropConnectLinear):
                assert layer.p == p