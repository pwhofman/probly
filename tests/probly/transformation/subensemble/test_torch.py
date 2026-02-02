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
            "torch_custom_model",
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

        subensemble_model = subensemble(model, num_heads=num_heads)

        count_linear_original = count_layers(model, nn.Linear)
        count_sequential_original = 1 if model_fixture == "torch_custom_model" else count_layers(model, nn.Sequential)
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_dropout_original = count_layers(model, nn.Dropout)

        count_sequential_subensemble = count_layers(subensemble_model, nn.Sequential)

        assert isinstance(subensemble_model, nn.ModuleList)
        assert len(subensemble_model) == num_heads
        assert count_sequential_subensemble == 3 * num_heads * count_sequential_original
        for member in subensemble_model:
            count_linear_subensemble = count_layers(member, nn.Linear)
            count_convolutional_subensemble = count_layers(member, nn.Conv2d)
            count_dropout_subensemble = count_layers(member, nn.Dropout)
            assert count_linear_subensemble == count_linear_original
            assert count_convolutional_subensemble == count_convolutional_original
            assert count_dropout_subensemble == count_dropout_original

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
        count_linear_original = count_layers(model, nn.Linear)
        count_sequential_original = 1 if model_fixture == "torch_custom_model" else count_layers(model, nn.Sequential)
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_dropout_original = count_layers(model, nn.Dropout)

        count_sequential_subensemble = count_layers(subensemble_model, nn.Sequential)

        assert isinstance(subensemble_model, nn.ModuleList)
        assert len(subensemble_model) == num_heads
        assert count_sequential_subensemble == 3 * num_heads * count_sequential_original
        for member in subensemble_model:
            count_linear_subensemble = count_layers(member, nn.Linear)
            count_convolutional_subensemble = count_layers(member, nn.Conv2d)
            count_dropout_subensemble = count_layers(member, nn.Dropout)
            assert count_linear_subensemble == count_linear_original
            assert count_convolutional_subensemble == count_convolutional_original
            assert count_dropout_subensemble == count_dropout_original

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
        count_linear_original = count_layers(model, nn.Linear)
        count_sequential_original = 1 if model_fixture == "torch_custom_model" else count_layers(model, nn.Sequential)
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_dropout_original = count_layers(model, nn.Dropout)

        count_sequential_subensemble = count_layers(subensemble_model, nn.Sequential)

        assert isinstance(subensemble_model, nn.ModuleList)
        assert len(subensemble_model) == num_heads
        assert count_sequential_subensemble == 3 * num_heads * count_sequential_original
        for member in subensemble_model:
            count_linear_subensemble = count_layers(member, nn.Linear)
            count_convolutional_subensemble = count_layers(member, nn.Conv2d)
            count_dropout_subensemble = count_layers(member, nn.Dropout)
            assert count_linear_subensemble == count_linear_original * 2
            assert count_convolutional_subensemble == count_convolutional_original * 2
            assert count_dropout_subensemble == count_dropout_original * 2


class TestParameterReset:
    """Tests for parameter resetting behavior in subensemble."""

    def test_parameter_reset(self, torch_model_small_2d_2d: nn.Module) -> None:
        """reset_params=True should initialize heads with different parameters."""
        num_heads = 2
        head_layer = 1

        subensemble_model = subensemble(
            torch_model_small_2d_2d,
            num_heads=num_heads,
            reset_params=True,
            head_layer=head_layer,
        )

        head_member1 = subensemble_model[0][-head_layer:]
        head_member2 = subensemble_model[1][-head_layer:]
        params1 = next(iter(head_member1.parameters())).detach().clone()
        params2 = next(iter(head_member2.parameters())).detach().clone()
        assert not torch.equal(params1, params2)

    def test_no_parameter_reset(
        self,
        torch_model_small_2d_2d: nn.Module,
    ) -> None:
        """reset_params=False should keep identical initialization across heads."""
        num_heads = 2
        head_layer = 1

        subensemble_model = subensemble(
            torch_model_small_2d_2d,
            num_heads=num_heads,
            reset_params=False,
            head_layer=head_layer,
        )
        head_member1 = subensemble_model[0][-head_layer:]
        head_member2 = subensemble_model[1][-head_layer:]
        params1 = next(iter(head_member1.parameters())).detach().clone()
        params2 = next(iter(head_member2.parameters())).detach().clone()
        assert torch.equal(params1, params2)


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
        head_layer = count_layers(torch_model_small_2d_2d, nn.Linear) + 1

        with pytest.raises(
            ValueError,
            match=f"head_layer {head_layer} must be less than to {head_layer - 1}",
        ):
            subensemble(
                torch_model_small_2d_2d,
                num_heads=num_heads,
                head_layer=head_layer,
            )
