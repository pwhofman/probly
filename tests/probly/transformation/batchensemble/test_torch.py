"""Test for torch batchensemble models."""

from __future__ import annotations

import pytest

from probly.layers.torch import BatchEnsembleConv2d, BatchEnsembleLinear
from probly.transformation.batchensemble import batchensemble
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestBatchEnsembleLayers:
    """Tests for BatchEnsembleLinear and BatchEnsembleConv2d layer generation."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "torch_model_small_2d_2d",
            "torch_conv_linear_model",
            "torch_regression_model_1d",
            "torch_regression_model_2d",
            "torch_dropout_model",
        ],
    )
    def test_batchensemble(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        model = request.getfixturevalue(model_fixture)
        num_members = 5

        batchensemble_model = batchensemble(model, num_members)

        count_linear_original = count_layers(model, nn.Linear)
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_sequential_original = count_layers(model, nn.Sequential)

        count_linear_modified = count_layers(batchensemble_model, nn.Linear)
        count_convolutional_modified = count_layers(batchensemble_model, nn.Conv2d)
        count_sequential_modified = count_layers(batchensemble_model, nn.Sequential)

        count_batchensemblelinear_modified = count_layers(batchensemble_model, BatchEnsembleLinear)
        count_batchensembleconv2d_modified = count_layers(batchensemble_model, BatchEnsembleConv2d)

        assert isinstance(batchensemble_model, nn.Sequential)
        assert count_batchensemblelinear_modified == count_linear_original
        assert count_batchensembleconv2d_modified == count_convolutional_original
        assert count_linear_modified == 0
        assert count_convolutional_modified == 0
        assert count_sequential_original == count_sequential_modified
