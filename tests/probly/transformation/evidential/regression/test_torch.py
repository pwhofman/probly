"""Test for torch evidential regression models."""

from __future__ import annotations

import pytest

from probly.layers.torch import NormalInverseGammaLinear
from probly.transformation.evidential.regression import evidential_regression
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if evidential regression correctly replaces the last Linear layer with NIG layer.

        This function verifies that:
        - The last Linear layer is replaced with NormalInverseGammaLinear.
        - The structure of the model remains unchanged except for the replaced layer.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the
            NIG layer is not inserted correctly.
        """
        model = evidential_regression(torch_model_small_2d_2d)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        # count number of NIG layers in original model
        count_nig_original = count_layers(
            torch_model_small_2d_2d,
            NormalInverseGammaLinear,
        )

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of NIG layers in modified model
        count_nig_modified = count_layers(model, NormalInverseGammaLinear)

        # check that one Linear was replaced with NIG
        assert model is not None
        assert isinstance(model, nn.Sequential)
        assert count_nig_modified == count_nig_original + 1
        assert count_linear_modified == count_linear_original - 1

    def test_convolutional_network(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests the convolutional neural network modification with NIG layer.

        This function evaluates whether the given convolutional neural network model
        has been correctly modified to replace the last Linear layer with a
        NormalInverseGammaLinear layer without altering the number of convolutional layers.

        Parameters:
            torch_conv_linear_model: The original convolutional neural network model to be tested.

        Raises:
            AssertionError: If the modified model deviates in structure other than
            the replacement of the last Linear layer.
        """
        model = evidential_regression(torch_conv_linear_model)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        # count number of NIG layers in original model
        count_nig_original = count_layers(
            torch_conv_linear_model,
            NormalInverseGammaLinear,
        )
        # count number of nn.Conv2d layers in original model
        count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of NIG layers in modified model
        count_nig_modified = count_layers(model, NormalInverseGammaLinear)
        # count number of nn.Conv2d layers in modified model
        count_conv_modified = count_layers(model, nn.Conv2d)

        # check that one Linear was replaced with NIG, convs unchanged
        assert model is not None
        assert isinstance(model, nn.Sequential)
        assert count_nig_modified == count_nig_original + 1
        assert count_linear_modified == count_linear_original - 1
        assert count_conv_modified == count_conv_original

    def test_custom_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the custom model modification with NIG layer.

        Parameters:
            torch_custom_model: A custom PyTorch model (not Sequential).

        Raises:
            AssertionError: If the modified model structure is not correct.
        """
        model = evidential_regression(torch_custom_model)

        # check that model contains NIG layer
        assert model is not None
        has_nig = any(isinstance(m, NormalInverseGammaLinear) for m in model.modules())
        assert has_nig, "NormalInverseGammaLinear layer should be present in the model"


class TestLayerReplacement:
    """Test class for layer replacement tests."""

    def test_nig_layer_present(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests that a NormalInverseGammaLinear layer is present after transformation.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested.

        Raises:
            AssertionError: If the NIG layer is not present.
        """
        model = evidential_regression(torch_model_small_2d_2d)

        # check that NIG layer is in the model
        has_nig = any(isinstance(m, NormalInverseGammaLinear) for m in model.modules())
        assert has_nig, "NormalInverseGammaLinear layer should be present in the model"

    def test_last_layer_is_nig(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests that the last layer of the modified model is NormalInverseGammaLinear.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested.

        Raises:
            AssertionError: If the last layer is not NIG.
        """
        model = evidential_regression(torch_model_small_2d_2d)

        # The model should have NIG as the last layer
        assert isinstance(model, nn.Sequential)
        # Get the last layer
        last_layer = list(model.children())[-1]
        assert isinstance(
            last_layer,
            NormalInverseGammaLinear,
        ), f"Last layer should be NormalInverseGammaLinear, but got {type(last_layer)}"
