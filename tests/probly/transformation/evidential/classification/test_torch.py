"""Test for torch evidential classification models."""

from __future__ import annotations

import pytest

from probly.transformation.evidential.classification import evidential_classification

from tests.probly.torch_utils import count_layers  

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_sequential_network(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if a sequential model appends an evidential classification layer correctly

        This function verifies that:
        - The number of linear layers remains unchanged after appending the evidential classification layer.
        - The model becomes a sequential model with the softplus activation layer appended at the end.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner
        """

        model = evidential_classification(torch_model_small_2d_2d)


        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)

        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)

        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)

        # check that the model is a sequential with appended evidential classification layer at the end
        assert model is not None
        assert isinstance(model, nn.Sequential)
        assert count_linear_modified == count_linear_original
        assert count_sequential_modified == count_sequential_original + 1

        # check if last layer is a Softplus activation
        last_layer = list(model.children())[-1]
        assert isinstance(last_layer, nn.Softplus)

    def test_non_sequential_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the non-sequential model modification with an appended evidential classification layer."""

        # check if model type is correct
        model = evidential_classification(torch_custom_model)
        assert isinstance(model, nn.Sequential)
        # check if last layer is a Softplus activation
        last_layer = list(model.children())[-1]
        assert isinstance(last_layer, nn.Softplus)
