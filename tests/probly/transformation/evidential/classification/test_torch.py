"""Test for torch evidential classification models."""

from __future__ import annotations

import pytest

from probly.transformation.evidential.classification import evidential_classification
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if evidential classification correctly appends Softplus activation to a linear model.

        This function verifies that:
        - A Softplus activation layer is added to the end of the model.
        - The structure of the model remains unchanged except for the added activation.
        - The model is wrapped in a Sequential container.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the
            Softplus layer is not appended correctly.
        """
        model = evidential_classification(torch_model_small_2d_2d)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        # count number of nn.Softplus layers in original model
        count_softplus_original = count_layers(torch_model_small_2d_2d, nn.Softplus)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of nn.Softplus layers in modified model
        count_softplus_modified = count_layers(model, nn.Softplus)

        # check that the model is not modified except for the softplus activation
        assert model is not None
        assert isinstance(model, nn.Sequential)
        assert count_softplus_modified == count_softplus_original + 1
        assert count_linear_modified == count_linear_original

        # Verify the model has exactly 2 children: original model + Softplus
        children = list(model.children())
        assert len(children) == 2
        assert isinstance(children[1], nn.Softplus)

    def test_convolutional_network(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests the convolutional neural network modification with appended Softplus activation.

        This function evaluates whether the given convolutional neural network model
        has been correctly modified to include a Softplus activation layer at the end
        without altering the number of other components such as linear, sequential,
        or convolutional layers.

        Parameters:
            torch_conv_linear_model: The original convolutional neural network model to be tested.

        Raises:
            AssertionError: If the modified model deviates in structure other than
            the addition of the Softplus activation or does not meet the expected constraints.
        """
        model = evidential_classification(torch_conv_linear_model)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        # count number of nn.Softplus layers in original model
        count_softplus_original = count_layers(torch_conv_linear_model, nn.Softplus)
        # count number of nn.Conv2d layers in original model
        count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of nn.Softplus layers in modified model
        count_softplus_modified = count_layers(model, nn.Softplus)
        # count number of nn.Conv2d layers in modified model
        count_conv_modified = count_layers(model, nn.Conv2d)

        # check that the model is not modified except for the softplus activation
        assert model is not None
        assert isinstance(model, nn.Sequential)
        assert count_softplus_modified == count_softplus_original + 1
        assert count_linear_modified == count_linear_original
        assert count_conv_modified == count_conv_original

    def test_custom_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the custom model modification with appended Softplus activation.

        Parameters:
            torch_custom_model: A custom PyTorch model (not Sequential).

        Raises:
            AssertionError: If the modified model is not wrapped correctly.
        """
        model = evidential_classification(torch_custom_model)

        # check if model is wrapped in Sequential
        assert model is not None
        assert isinstance(model, nn.Sequential)

        # Verify structure: should have 2 children (original + Softplus)
        children = list(model.children())
        assert len(children) == 2
        assert isinstance(children[1], nn.Softplus)


class TestActivationFunction:
    """Test class for activation function tests."""

    def test_softplus_appended(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests that a Softplus activation is appended to the model.

        This function verifies that after applying evidential_classification,
        the model contains exactly one more Softplus layer than the original.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested.

        Raises:
            AssertionError: If the Softplus layer is not appended correctly.
        """
        model = evidential_classification(torch_model_small_2d_2d)

        # count Softplus layers before and after
        count_softplus_original = count_layers(torch_model_small_2d_2d, nn.Softplus)
        count_softplus_modified = count_layers(model, nn.Softplus)

        assert count_softplus_modified == count_softplus_original + 1

    def test_last_layer_is_softplus(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests that the last layer of the modified model is Softplus.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested.

        Raises:
            AssertionError: If the last layer is not Softplus.
        """
        model = evidential_classification(torch_model_small_2d_2d)

        # The model should be a Sequential with the last layer being Softplus
        assert isinstance(model, nn.Sequential)
        # Get the last layer
        last_layer = list(model.children())[-1]
        assert isinstance(last_layer, nn.Softplus), f"Last layer should be Softplus, but got {type(last_layer)}"

    def test_softplus_is_present(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests that Softplus activation is present in various network types.

        Parameters:
            torch_conv_linear_model: A convolutional neural network model.

        Raises:
            AssertionError: If Softplus is not found in the model.
        """
        model = evidential_classification(torch_conv_linear_model)

        # check that Softplus is in the model
        has_softplus = any(isinstance(m, nn.Softplus) for m in model.modules())
        assert has_softplus, "Softplus activation should be present in the model"
