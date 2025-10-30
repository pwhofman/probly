"""Test for torch dropout models."""

from __future__ import annotations

import pytest


from probly.transformation import evidential_classification
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network_with_first_linear(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if a model incorporates a dropout layer correctly when a linear layer succeeds it.

        This function verifies that:
        - A dropout layer is added before each linear layer in the model, except for the last linear layer.
        - The structure of the model remains unchanged except for the added dropout layers.
        - Only the specified probability parameter is applied in dropout modifications.

        It performs counts and asserts to ensure the modified model adheres to expectations.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the dropout layer is not
            inserted correctly after linear layers.
        """
        
        model = evidential_classification(torch_model_small_2d_2d)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        # count number of nn.Softplus layers in original model
        count_Softplus_original = count_layers(torch_model_small_2d_2d, nn.Softplus)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of nn.Dropout layers in modified model
        count_dropout_modified = count_layers(model, nn.Softplus)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)

        # check that the model is not modified except for the dropout layer
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert (count_linear_original - 1) == count_dropout_modified
        assert count_linear_modified == count_linear_original
        assert count_Softplus_original == 0
        assert count_sequential_original == count_sequential_modified

    def test_convolutional_network(self, torch_conv_linear_model: nn.Sequential) -> None:
    
        model = evidential_classification(torch_conv_linear_model)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        # count number of nn.Dropout layers in original model
        count_dropout_original = count_layers(torch_conv_linear_model, nn.Softplus)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
        # count number of nn.Conv2d layers in original model
        count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of nn.Dropout layers in modified model
        count_dropout_modified = count_layers(model, nn.Softplus)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)
        # count number of nn.Conv2d layers in modified model
        count_conv_modified = count_layers(model, nn.Conv2d)

        # check that the model is not modified except for the dropout layer
        assert model is not None
        assert isinstance(model, type(torch_conv_linear_model))
        assert count_linear_original == count_dropout_modified
        assert count_linear_original == count_linear_modified
        assert count_dropout_original == 0
        assert count_sequential_original == count_sequential_modified
        assert count_conv_original == count_conv_modified

    def test_custom_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the custom model modification with added dropout layers."""
        model = evidential_classification(torch_custom_model)

        # check if model type is correct
        assert isinstance(model, type(torch_custom_model))
        assert not isinstance(model, nn.Sequential)

    @pytest.mark.skip(reason="Not yet implemented in probly")
    def test_dropout_model(self, torch_evidential_classification_model: nn.Sequential) -> None:
        """Tests the dropout model modification if dropout already exists."""
        model = evidential_classification(torch_evidential_classification_model)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_evidential_classification_model, nn.Linear)
        # count number of nn.Dropout layers in original model
        count_dropout_original = count_layers(torch_evidential_classification_model, nn.Softplus)
        # count number of nn.Dropout layers in modified model
        count_dropout_modified = count_layers(model, nn.Softplus)

        # check that model has no duplicate dropout layers
        assert count_dropout_original == 1
        assert count_linear_original == 2
        assert (count_linear_original - 1) == count_dropout_modified

       

