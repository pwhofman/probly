"""Tests for the evidential classification transformation using PyTorch."""

from __future__ import annotations

import pytest

from probly.transformation.evidential import classification
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")  # ensure torch is available for the next import
from torch import nn  # noqa: E402


class TestEvidentialTorchAppender:
    def test_sequential_appends_softplus(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Test if nn.Softplus is appended and checks if structure is being held except appended function.

        This function verifies that:
        - A Softplus layer is added after each nn.Module layer in the model, except for the last layer.
        - The structure of the model remains unchanged except for the added Softplus layers.

        It performs counts and asserts to ensure the modified model adheres to expectations.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the Sofplus layer is not
            inserted correctly after nn.Module layers
        """
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
        assert isinstance(model, nn.Sequential)
        assert (count_module_original + 2) == count_module_modified
        assert count_softplus_modified == 1
        assert count_softplus_original == 0
        assert (count_sequential_original + 1) == count_sequential_modified

    def test_convolutional_network(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests the convolutional neural network modification with added activation function at the end of modules.

        This function evaluates whether the given convolutional neural network model
        has been correctly modified to include Softplus layers without altering the
        number of other components such as linear, sequential, or convolutional layers.

        Parameters:
            torch_conv_linear_model: The original convolutional neural network model to be tested.

        Raises:
            AssertionError: If the modified model deviates in structure other than
            the addition of Softplus layers or does not meet the expected constraints.
        """
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
        """Tests the custom model modification with appended activation function.

        This function checks whether a custom PyTorch model has been correctly modified
        to include a Softplus activation function at the end.

        Parameters:
            torch_custom_model: The custom PyTorch model to be tested.

        Raises:
            AssertionError: If the modified model does not have the expected structure
        """
        model = classification.evidential_classification(torch_custom_model)

        # check if model type is correct
        assert isinstance(model, nn.Sequential)
        assert model[0] is torch_custom_model
        assert isinstance(model[1], nn.Softplus)

    @pytest.mark.skip(reason="Not yet implemented in probly")
    def test_evidential_classification_model(self, torch_evidential_classification_model: nn.Sequential) -> None:
        """Tests the evidential classification model modification if Softplus already exists.

        This function verifies that when an evidential classification model already contains
        a Softplus layer, the modification process does not introduce duplicate Softplus layers.

        Parameters:
            torch_evidential_classification_model: The evidential classification model to be tested.

        Raises:
            AssertionError: If duplicate Softplus layers are found in the modified model.
        """
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

    def test_forward_passes_for_shapes(self) -> None:
        """Tests if forward passes work for different input shapes for 2d and 3d models."""
        # test for 2D input
        model_2d = classification.evidential_classification(
            nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3)),
        )
        input_2d = torch.rand(4, 10)
        output_2d = model_2d(input_2d)
        assert output_2d.shape == (4, 3)

        # test for 3D input
        model_3d = classification.evidential_classification(
            nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2)),
        )
        input_3d = torch.rand(6, 7, 8)
        output_3d = model_3d(input_3d)
        assert output_3d.shape == (6, 7, 2)

    def test_grads_are_kept(self) -> None:
        """Tests if gradients are kept after appending the activation function."""
        model = classification.evidential_classification(
            nn.Sequential(nn.Linear(5, 3), nn.Linear(3, 2)),
        )
        input_tensor = torch.rand(2, 5, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape, (
            f"Gradient shape mismatch. Expected {input_tensor.shape}, but output is {input_tensor.grad.shape}"
        )

    def test_datatypes_preserved(self) -> None:
        """Tests if the data type is preserved after appending the activation function."""
        model = classification.evidential_classification(
            nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 2)),
        )
        expected_dtype = torch.float32
        input_tensor = torch.rand(3, 4, dtype=expected_dtype)
        output = model(input_tensor)

        assert output.dtype == expected_dtype, (
            f"Datatype not preserved. Expected dtype is {expected_dtype}, but output is {output.dtype}"
        )
