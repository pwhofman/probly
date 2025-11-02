"""Test for flax dropconnect models."""

from __future__ import annotations

import pytest

from probly.transformation import dropconnect
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network_with_first_linear(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if a model incorporates DropConnect layers correctly when replacing linear layers.

        This function verifies that:
        - Linear layers are replaced by DropConnect layers (except first layer)
        - The structure of the model remains unchanged except for the layer replacements
        - Only the specified probability parameter is applied in dropconnect modifications.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested, specified as a sequential model.  # ← torch → flax

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if 
            the DropConnect layer is not inserted correctly.
        """
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        
        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        # count number of nnx.DropConnect layers in original model
        count_dropconnect_original = count_layers(flax_model_small_2d_2d, nnx.DropConnect)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_model_small_2d_2d, nnx.Sequential)

        # count number of nnx.Linear layers in modified model
        count_linear_modified = count_layers(model, nnx.Linear)
        # count number of nnx.DropConnect layers in modified model
        count_dropconnect_modified = count_layers(model, nnx.DropConnect)
        # count number of nnx.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nnx.Sequential)
        
        # check that linear layers are replaced by DropConnectLinear layers 
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert count_dropconnect_modified == count_linear_original - 1  
        assert count_linear_modified == 1 
        assert count_dropconnect_original == 0
        assert count_sequential_original == count_sequential_modified
        
    def test_convolutional_network(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests the convolutional neural network modification with DropConnect layers.

        This function evaluates whether the given convolutional neural network model
        has been correctly modified to replace linear layers with DropConnect layers
        without altering other components.

        Parameters:
            flax_conv_linear_model: The original convolutional neural network model to be tested.  # ← torch → flax

        Raises:
            AssertionError: If the modified model deviates in structure other than
            the replacement of linear layers.
        """
        p = 0.5
        model = dropconnect(flax_conv_linear_model, p)
        
        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_conv_linear_model, nnx.Sequential)
        # count number of nnx.Conv2d layers in original model
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv2d)
        # count number of nnx.DropConnect layers in original model
        count_dropconnect_original = count_layers(flax_conv_linear_model, nnx.DropConnect)

        # count number of nnx.DropConnect layers in modified model
        count_dropconnect_modified = count_layers(model, nnx.DropConnect)
        # count number of nnx.Linear layers in modified model
        count_linear_modified = count_layers(model, nnx.Linear)
        # count number of nnx.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nnx.Sequential)
        # count number of nnx.Conv2d layers in modified model
        count_conv_modified = count_layers(model, nnx.Conv2d)

        # check that the model is not modified except for the dropout layer
        assert model is not None
        assert isinstance(model, type(flax_conv_linear_model))
        assert count_dropconnect_original == 0 
        assert count_dropconnect_modified == count_linear_original  # No skipping in conv models
        assert count_linear_modified == 0  # All linear layers should be replaced
        assert count_sequential_original == count_sequential_modified
        assert count_conv_original == count_conv_modified
            
        # Additional verification: the linear layer should be replaced with DropConnectLinear
        assert isinstance(model[3], nnx.DropConnect), "Linear layer should be replaced with DropConnectLinear"
        
    def test_custom_network(self, flax_custom_model: nnx.Module) -> None:
        """Tests the custom model modification with DropConnect layers."""
        p = 0.5
        model = dropconnect(flax_custom_model, p)

        # check if model type is correct
        assert isinstance(model, type(flax_custom_model))
        assert not isinstance(model, nnx.Sequential)

    def test_first_layer_skipping(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Test that the first linear layer is not replaced due to is_first_layer condition."""
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)
        
        # First layer should stay unchanged
        assert isinstance(model[0], nnx.Linear), "First layer should remain nn.Linear"
        assert not isinstance(model[0], nnx.DropConnect), "First layer should not be DropConnectLinear"
        
        # Linear layers should be replaced
        for i in range(1, len(model)):
            if isinstance(flax_model_small_2d_2d[i], nnx.Linear):
                assert isinstance(model[i], nnx.DropConnect), \
                    f"Layer {i} should be DropConnectLinear"
                    
    def test_dropconnect_duplicates_behavior(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests that reapplying in dropconnect doesn't create duplicates."""
        
        first_p = 0.3
        model_after_first = dropconnect(flax_model_small_2d_2d, first_p)
        
        second_p = 0.5
        model_after_second = dropconnect(model_after_first, second_p)
        
        count_dropconnect_first = count_layers(model_after_first, nnx.DropConnect)
        count_linear_first = count_layers(model_after_first, nnx.Linear)
        
        count_dropconnect_second = count_layers(model_after_second, nnx.DropConnect)
        count_linear_second = count_layers(model_after_second, nnx.Linear)
        
        # Layers are replaced, reapplying should not change layer counts
        assert count_dropconnect_first == count_dropconnect_second
        assert count_linear_first == count_linear_second       

class TestPValues:
    """Test class for p-value tests with DropConnect."""

    def test_linear_network_p_value(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests the DropConnect layer's p-value in a given neural network model.

        This function verifies that DropConnect layers inside the provided neural network
        model have the expected p-value after applying the dropconnect transformation.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested for integration  # ← torch → flax

        Raises:
            AssertionError: If the p-value in a DropConnect layer does not match the expected value.
        """
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # check p value in DropConnect layers (skip first layer)
        for m in model.modules():
            if isinstance(m, nnx.DropConnect):
                assert m.p == p

    def test_conv_network_p_value(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests whether DropConnect layers in the convolutional model have the correct probability value.

        Arguments:
            flax_conv_linear_model: A sequential model containing convolutional and linear layers.  # ← torch → flax

        Raises:
            AssertionError: If the probability value in any DropConnect layer does not match the expected value.
        """
        p = 0.2
        model = dropconnect(flax_conv_linear_model, p)

        # check p value in DropConnect layers (skip first layer)
        for m in model.modules():
            if isinstance(m, nnx.DropConnect):
                assert m.p == p
        
        