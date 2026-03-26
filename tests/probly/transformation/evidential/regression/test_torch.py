"""Tests for torch evidential regression models."""

from __future__ import annotations

import pytest
import torch as th
from torch import nn

from probly.layers.torch import NormalInverseGammaLinear
from probly.transformation.evidential.regression import evidential_regression
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")


class TestEvidentialRegression:
    """Test class for torch evidential regression models."""

    def test_returns_a_clone(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if evidential_regression returns a clone of the input model."""
        original_model = torch_model_small_2d_2d

        new_model = evidential_regression(original_model)

        assert new_model is not original_model

    def test_replaces_only_last_linear_layer(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if evidential_regression *only* replaces the last linear layer.

        This function verifies that the new model has exactly one LESS nn.Linear layer
        than the original, and one NormalInverseGammaLinear (NIG) layer.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested.
        """
        original_model = torch_model_small_2d_2d
        new_model = evidential_regression(original_model)

        # Layer Count Checks
        count_linear_original = count_layers(original_model, nn.Linear)
        count_linear_modified = count_layers(new_model, nn.Linear)

        # Check the core logic: The new model should have one LESS nn.Linear layer
        assert count_linear_modified == (count_linear_original - 1)

        # The modified model should have exactly one NIG layer
        count_nig_modified = count_layers(new_model, NormalInverseGammaLinear)
        assert count_nig_modified == 1

    def test_last_layer_replacement_and_integrity(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests replacement of the last layer and verifies model integrity.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested.
        """
        original_model = torch_model_small_2d_2d

        # 1. Forward Pass & Shape Check
        input_data = th.randn(1, 2)

        # The output of the original model (Tensor) is needed to check the shape
        original_output = original_model(input_data)
        expected_output_shape = original_output.shape

        # Transformation
        new_model = evidential_regression(original_model)

        # 1a. Shape Check
        new_output = new_model(input_data)
        # KORREKTUR: Der NIG-Layer gibt ein Dictionary zurück, wir prüfen die Shape des 'gamma'-Tensors (mean).
        assert new_output["gamma"].shape == expected_output_shape

        # 1b. Last Layer Replacement Check
        # Check if the last layer in the new model is the NIG layer.
        last_layer_index = len(original_model) - 1
        last_layer_modified = new_model[last_layer_index]

        assert isinstance(last_layer_modified, NormalInverseGammaLinear)

        # 1c. Rest Model Integrity Check
        # The layers BEFORE the last layer must retain their original type.
        for i in range(last_layer_index):
            original_layer = original_model[i]
            modified_layer = new_model[i]
            assert type(original_layer) is type(modified_layer)
