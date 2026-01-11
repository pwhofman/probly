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
    def test_batchensemble_generation(
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
    def test_batchensemble_shapes(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        model = request.getfixturevalue(model_fixture)
        num_members = 5
        batchensemble_model = batchensemble(model, num_members)

        for fixture_layer, batchensemble_layer in zip(model.children(), batchensemble_model.children(), strict=False):
            if isinstance(fixture_layer, nn.Linear) and isinstance(batchensemble_layer, BatchEnsembleLinear):
                fixture_in_features = fixture_layer.in_features
                fixture_out_features = fixture_layer.out_features

                batchensemble_in_features = batchensemble_layer.in_features
                batchensemble_out_features = batchensemble_layer.out_features
                batchensemble_layer_s = batchensemble_layer.s.shape
                batchensemble_layer_r = batchensemble_layer.r.shape

                # assert if in_features and out_features are the same
                assert batchensemble_in_features == fixture_in_features
                assert batchensemble_out_features == fixture_out_features

                # assert if the shapes are correct
                assert batchensemble_layer_s == (num_members, batchensemble_in_features)
                assert batchensemble_layer_r == (num_members, batchensemble_out_features)

            if isinstance(fixture_layer, nn.Conv2d) and isinstance(batchensemble_layer, BatchEnsembleConv2d):
                fixture_in_channels = fixture_layer.in_channels
                fixture_out_channels = fixture_layer.out_channels

                batchensemble_in_channels = batchensemble_layer.in_channels
                batchensemble_out_channels = batchensemble_layer.out_channels
                batchensemble_layer_s = batchensemble_layer.s.shape
                batchensemble_layer_r = batchensemble_layer.r.shape

                # assert if in_channels and out_channels are the same
                assert batchensemble_in_channels == fixture_in_channels
                assert batchensemble_out_channels == fixture_out_channels

                # assert if the shapes are correct
                assert batchensemble_layer_s == (num_members, batchensemble_in_channels)
                assert batchensemble_layer_r == (num_members, batchensemble_out_channels)
