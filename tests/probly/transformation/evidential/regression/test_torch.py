"""Test for torch evidential regression models."""

from __future__ import annotations

import pytest

from probly.transformation.evidential.regression import evidential_regression

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestEvidentialRegression:
    """Test class for torch evidential regression models."""

    def test_returns_a_clone(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if evidential_regression returns a clone of the input model."""
        original_model = torch_model_small_2d_2d

        new_model = evidential_regression(original_model)

        assert new_model is not original_model
