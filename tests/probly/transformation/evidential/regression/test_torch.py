"""Test for torch evidential regression models."""

from __future__ import annotations

import pytest

from probly.layers.torch import NormalInverseGammaLinear
from probly.transformation.evidential.regression import evidential_regression
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestEvidentialRegression:
    """Test class for torch evidential regression models."""

    def test_returns_a_clone(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if evidential_regression returns a clone of the input model."""
        original_model = torch_model_small_2d_2d

        new_model = evidential_regression(original_model)

        assert new_model is not original_model

    def test_replaces_only_last_linear_layer(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if evidential_regression *only* replaces the last linear layer.

        This function verifies that:
        - The structure of the model remains unchanged *except* for the last linear layer.
        - The last linear layer is replaced by a NormalInverseGammaLinear layer.
        - All other layers (including other nn.Linear layers) remain untouched.

        Parameters:
            torch_model_small_2d_2n: The torch model to be tested.

        Raises:
            AssertionError: If the structure of the model differs or the
            layers are not replaced correctly.
        """
        original_model = torch_model_small_2d_2d

        new_model = evidential_regression(original_model)

        #  --- Count layers in ORIGINAL model ---
        count_linear_original = count_layers(original_model, nn.Linear)
        count_nig_original = count_layers(original_model, NormalInverseGammaLinear)

        #  --- Count layers in NEW (modified) model ---
        count_linear_modified = count_layers(new_model, nn.Linear)
        count_nig_modified = count_layers(new_model, NormalInverseGammaLinear)

        #  --- Check model integrity ---
        assert new_model is not None
        assert isinstance(new_model, type(original_model))

        #  --- Check the core logic of the replacement ---

        # The original model should have no NIG layers
        assert count_nig_original == 0

        # The modified model should have exactly one NIG layer
        assert count_nig_modified == 1

        # The new model should have one LESS nn.Linear layer than the original
        assert count_linear_modified == (count_linear_original - 1)
