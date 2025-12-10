"""Test for torch subensemble models."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from probly.transformation.subensemble import subensemble
from tests.probly.torch_utils import count_layers


class TestGeneration:
    """Tests for different subensemble generations."""

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
    def test_subensemble_default(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        """Test for default subensemble generation."""
        model = request.getfixturevalue(model_fixture)
        num_heads = 5
        head_layer = 1  # default

        subensemble_model = subensemble(model, num_heads=num_heads)
        backbone = subensemble_model[0]
        heads = subensemble_model[1]

        count_linear_original = count_layers(model, nn.Linear)
        count_linear_backbone = count_layers(backbone, nn.Linear)
        count_linear_heads = count_layers(heads, nn.Linear)
        count_sequential_original = count_layers(model, nn.Sequential)
        count_sequential_backbone = count_layers(backbone, nn.Sequential)
        count_sequential_heads = count_layers(heads, nn.Sequential)
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_convolutional_backbone = count_layers(backbone, nn.Conv2d)
        count_convolutional_heads = count_layers(heads, nn.Conv2d)

        assert isinstance(subensemble_model, nn.Module)
        assert isinstance(backbone, nn.Sequential)
        assert isinstance(heads, nn.ModuleList)
        assert len(heads) == num_heads
        assert count_sequential_heads == num_heads
        assert count_linear_backbone == count_linear_original - head_layer
        assert (count_linear_original + num_heads - 1) == (count_linear_heads + count_linear_backbone)
        assert count_sequential_backbone == count_sequential_original
        assert count_convolutional_backbone == count_convolutional_original
        assert count_convolutional_heads == 0

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
    def test_subensemble_2_head_layers(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        """Test for 2 head layers subensemble generation."""
        model = request.getfixturevalue(model_fixture)
        num_heads = 5
        head_layer = 2

        subensemble_model = subensemble(
            model,
            num_heads=num_heads,
            head_layer=head_layer,
        )
        backbone = subensemble_model[0]
        heads = subensemble_model[1]

        count_linear_original = count_layers(model, nn.Linear)
        count_linear_backbone = count_layers(backbone, nn.Linear)
        count_linear_heads = count_layers(heads, nn.Linear)
        count_sequential_original = count_layers(model, nn.Sequential)
        count_sequential_backbone = count_layers(backbone, nn.Sequential)
        count_sequential_heads = count_layers(heads, nn.Sequential)
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_convolutional_backbone = count_layers(backbone, nn.Conv2d)
        count_convolutional_heads = count_layers(heads, nn.Conv2d)

        assert isinstance(subensemble_model, nn.Module)
        assert isinstance(backbone, nn.Sequential)
        assert isinstance(heads, nn.ModuleList)
        assert len(heads) == num_heads
        assert count_sequential_heads == num_heads
        if isinstance(model[-2], nn.Linear):
            assert count_linear_heads == num_heads * head_layer
            assert count_linear_backbone == count_linear_original - head_layer
            assert count_linear_original == head_layer + count_linear_backbone
        else:
            assert count_linear_heads == num_heads
            assert count_linear_backbone == count_linear_original - 1
        assert count_sequential_backbone == count_sequential_original
        assert count_convolutional_backbone == count_convolutional_original
        assert count_convolutional_heads == 0

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
    def test_subensemble_with_head_model(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        """Test for backbone and head model subensemble generation."""
        model = request.getfixturevalue(model_fixture)
        num_heads = 5

        subensemble_model = subensemble(
            base=model,
            num_heads=num_heads,
            head=model,
        )
        backbone = subensemble_model[0]
        heads = subensemble_model[1]

        count_linear_original = count_layers(model, nn.Linear)
        count_linear_backbone = count_layers(backbone, nn.Linear)
        count_linear_heads = count_layers(heads, nn.Linear)
        count_sequential_original = count_layers(model, nn.Sequential)
        count_sequential_backbone = count_layers(backbone, nn.Sequential)
        count_sequential_heads = count_layers(heads, nn.Sequential)
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_convolutional_backbone = count_layers(backbone, nn.Conv2d)
        count_convolutional_heads = count_layers(heads, nn.Conv2d)

        assert isinstance(subensemble_model, nn.Module)
        assert isinstance(backbone, nn.Sequential)
        assert isinstance(heads, nn.ModuleList)
        assert len(heads) == num_heads
        assert count_sequential_heads == num_heads
        assert count_linear_backbone == count_linear_original
        assert count_convolutional_backbone == count_convolutional_original
        assert count_sequential_backbone == count_sequential_original
        assert count_linear_heads == count_linear_original * num_heads
        assert count_convolutional_heads == count_convolutional_original * num_heads
        assert count_sequential_heads == num_heads
        assert count_linear_heads + count_linear_backbone == count_linear_original * (num_heads + 1)


class TestParameterReset:
    """Tests for parameter resetting behavior in subensemble."""

    def test_parameter_reset(self, torch_model_small_2d_2d: nn.Module) -> None:
        """reset_params=True should initialize heads with different parameters."""
        num_heads = 2

        subensemble_model = subensemble(
            torch_model_small_2d_2d,
            num_heads=num_heads,
            reset_params=True,
        )
        _, heads = subensemble_model

        params0 = next(iter(heads[0].parameters())).detach().clone()
        params1 = next(iter(heads[1].parameters())).detach().clone()

        # different memory & different values
        assert params0.data_ptr() != params1.data_ptr()
        assert not torch.equal(params0, params1)

    def test_no_reset_keeps_same_initialization(
        self,
        torch_model_small_2d_2d: nn.Module,
    ) -> None:
        """reset_params=False should keep identical initialization across heads."""
        num_heads = 2

        subensemble_model = subensemble(
            torch_model_small_2d_2d,
            num_heads=num_heads,
            reset_params=False,
        )
        _, heads = subensemble_model

        params0 = next(iter(heads[0].parameters())).detach().clone()
        params1 = next(iter(heads[1].parameters())).detach().clone()

        assert torch.equal(params0, params1)


class TestEdgeCases:
    """Tests for edge-case configurations of subensemble."""

    def test_invalid_head_layer(
        self,
        torch_model_small_2d_2d: nn.Module,
    ) -> None:
        """Test if head_layer <= 0 raises ValueError."""
        num_heads = 3

        with pytest.raises(
            ValueError,
            match="head_layer must be a positive number, but got head_layer=0 instead",
        ):
            subensemble(
                torch_model_small_2d_2d,
                num_heads=num_heads,
                head_layer=0,
            )

    def test_large_head_layer(
        self,
        torch_model_small_2d_2d: nn.Module,
    ) -> None:
        """Test if backbone can be empty while head is an ensemble of the base model."""
        num_heads = 2
        head_layer = count_layers(torch_model_small_2d_2d, nn.Linear)

        subensemble_model = subensemble(
            torch_model_small_2d_2d,
            num_heads=num_heads,
            head_layer=head_layer,
        )
        backbone = subensemble_model[0]
        heads = subensemble_model[1]

        original_layers = count_layers(torch_model_small_2d_2d, nn.Module)
        backbone_layers = count_layers(backbone, nn.Module)

        assert isinstance(subensemble_model, nn.Module)
        assert isinstance(backbone, nn.Sequential)
        assert isinstance(heads, nn.ModuleList)
        assert len(heads) == num_heads
        assert backbone_layers == 1  # Empty Sequential
        for head in heads:
            head_layers = count_layers(head, nn.Module)
            assert head_layers == original_layers
