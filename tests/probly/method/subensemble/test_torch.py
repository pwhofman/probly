"""Test for torch subensemble models."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from probly.method.subensemble import subensemble
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

        subensemble_model = subensemble(model, num_heads=num_heads)

        count_linear_original = count_layers(model, nn.Linear)
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_dropout_original = count_layers(model, nn.Dropout)

        assert isinstance(subensemble_model, nn.ModuleList)
        assert len(subensemble_model) == num_heads
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
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_dropout_original = count_layers(model, nn.Dropout)

        assert isinstance(subensemble_model, nn.ModuleList)
        assert len(subensemble_model) == num_heads
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
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_dropout_original = count_layers(model, nn.Dropout)

        assert isinstance(subensemble_model, nn.ModuleList)
        assert len(subensemble_model) == num_heads
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

    def test_non_sequential_without_head_raises(self) -> None:
        """Non-Sequential base without an explicit head should error and point to the workaround."""

        class NotSequential(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        with pytest.raises(ValueError, match="pass an explicit head module"):
            subensemble(NotSequential(), num_heads=2)


class TestNonSequentialBackbone:
    """Tests for the explicit-head path on non-Sequential backbones (the ResNet-style use case)."""

    def test_structure_with_explicit_head(self, torch_tiny_encoder: nn.Module) -> None:
        """Each member is Sequential(frozen_backbone, head_i) with len == num_heads."""
        head = nn.Linear(4, 3)
        num_heads = 4

        model = subensemble(torch_tiny_encoder, num_heads=num_heads, head=head)

        assert isinstance(model, nn.ModuleList)
        assert len(model) == num_heads
        for member in model:
            assert isinstance(member, nn.Sequential)

    def test_forward_runs(self, torch_tiny_encoder: nn.Module) -> None:
        """Forward through a member produces the head's output shape."""
        head = nn.Linear(4, 3)
        model = subensemble(torch_tiny_encoder, num_heads=2, head=head)

        x = torch.randn(5, 1, 8, 8)
        out = model[0](x)

        assert out.shape == (5, 3)

    def test_heads_have_independent_init_when_reset(self, torch_tiny_encoder: nn.Module) -> None:
        """reset_params=True yields different head weights across members."""
        head = nn.Linear(4, 3)
        model = subensemble(torch_tiny_encoder, num_heads=2, head=head, reset_params=True)

        w0 = model[0][1].weight.detach().clone()
        w1 = model[1][1].weight.detach().clone()
        assert not torch.equal(w0, w1)

    def test_backbone_params_frozen(self, torch_tiny_encoder: nn.Module) -> None:
        """All backbone parameters have requires_grad=False after wrapping."""
        head = nn.Linear(4, 3)
        model = subensemble(torch_tiny_encoder, num_heads=2, head=head)

        frozen_backbone = model[0][0]
        for p in frozen_backbone.parameters():
            assert p.requires_grad is False
        head_params = list(model[0][1].parameters())
        assert all(p.requires_grad for p in head_params)


class TestFrozenBackboneInvariants:
    """Regression tests for the eval-mode-stable behavior added by _FrozenBackbone."""

    def test_backbone_stays_eval_after_model_train(self) -> None:
        """model.train() must not flip the backbone (or its BN children) back to train mode."""
        backbone = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4))
        head = nn.Linear(4, 2)
        model = subensemble(backbone, num_heads=2, head=head)

        model.train()

        for member in model:
            frozen_backbone = member[0]
            assert frozen_backbone.training is False
            for sub in frozen_backbone.modules():
                assert sub.training is False
            assert member[1].training is True

    def test_bn_running_stats_dont_drift(self) -> None:
        """Forward through a member must not update the backbone's BatchNorm running stats."""
        bn = nn.BatchNorm2d(4)
        running_mean = bn.running_mean
        running_var = bn.running_var
        assert running_mean is not None
        assert running_var is not None
        running_mean.fill_(0.5)
        running_var.fill_(2.0)
        backbone = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), bn)
        head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))

        model = subensemble(backbone, num_heads=1, head=head)
        model.train()

        running_mean_before = running_mean.detach().clone()
        running_var_before = running_var.detach().clone()
        model[0](torch.randn(8, 1, 8, 8))

        assert torch.equal(running_mean, running_mean_before)
        assert torch.equal(running_var, running_var_before)
