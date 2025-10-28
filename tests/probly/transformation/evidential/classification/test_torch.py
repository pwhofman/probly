from __future__ import annotations
import pytest

from probly.transformation.evidential import classification
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")
from torch import nn

class TestEvidentialTorchAppender:
    def test_sequential_appends_softplus(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Test if nn.Softplus (the activation function) is appended, checks if structure is being held except appended function"""

        model = classification.evidential_classification(torch_model_small_2d_2d)

        # count number of nn.Module layers in original model
        count_module_original = count_layers(torch_model_small_2d_2d, nn.Module)
        # count number of nn.Softplus layers in original model
        count_softplus_original = count_layers(torch_model_small_2d_2d, nn.Softplus)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

        # count number of nn.Module layers in modified model
        count_module_modified = count_layers(model, nn.Module)
        # count number of nn.Softplus layers in modified model
        count_softplus_modified = count_layers(model, nn.Softplus)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)

        # check that model structure is not modified except appended nn.Softplus (activation function)
        assert model is not None
        assert isinstance (model, nn.Sequential)
        assert (count_module_original + 2) == count_module_modified
        assert count_softplus_modified == 1
        assert count_softplus_original == 0
        assert (count_sequential_original + 1) == count_sequential_modified
    
    def test_convolutional_network (self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests the convolutional neural network modification with added activation function at the end of modules"""
        model = classification.evidential_classification(torch_conv_linear_model)

        # count number of nn.Module layers in original model
        count_module_original = count_layers(torch_conv_linear_model, nn.Module)
        # count number of nn.Softplus layers in original model
        count_softplus_original = count_layers(torch_conv_linear_model, nn.Softplus)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
        # count number of nn.Conv2d layers in original model
        count_conv2d_original = count_layers(torch_conv_linear_model, nn.Conv2d)

        # count number of nn.Module layers in modified model
        count_module_modified = count_layers(model, nn.Module)
        # count number of nn.Softplus layers in modified model
        count_softplus_modified = count_layers(model, nn.Softplus)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)
        # count number of nn.Conv2d layers in original model
        count_conv2d_modified = count_layers(model, nn.Conv2d)

        # check that model structure is not modified except appended nn.Softplus (activation function) 
        assert model is not None
        assert isinstance(model, nn.Sequential)
        assert (count_softplus_original + 1) == count_softplus_modified
        assert count_softplus_original == 0
        assert (count_sequential_original + 1) == count_sequential_modified
        assert (count_module_original + 2) == count_module_modified
        assert count_conv2d_original == count_conv2d_modified
    
    def test_custom_model(self, torch_custom_model: nn.Module) -> None:
        """Tests the custom model modification with appended activation function"""
        model = classification.evidential_classification(torch_custom_model)

        # check if model type is correct
        assert isinstance(model, nn.Sequential)
        assert model[0] is torch_custom_model
        assert isinstance(model[1], nn.Softplus)

    @pytest.mark.skip(reason="Not yet implemented in probly")
    def test_evidential_classification_model(self, torch_evidential_classification_model: nn.Sequential) -> None:
        """Tests the evidential classification model modification if Softplus already exists"""
        model = classification.evidential_classification(torch_evidential_classification_model)

        # count number of nn.Module layers in original model
        count_module_original = count_layers(torch_evidential_classification_model, nn.Module)
        # count number of nn.Softplus layers in original model
        count_softplus_original = count_layers(torch_evidential_classification_model, nn.Softplus)
        # count number of nn.Softplus layers in modified model
        count_softplus_modified = count_layers(model, nn.Softplus)

        # check that model has no duplicate softplus layers
        assert count_softplus_original == 0
        assert count_module_original == 2
        assert count_softplus_modified == 1