from __future__ import annotations

import pytest
import torch
from torch import nn

from probly.transformation.subensemble import subensemble
from tests.probly.torch_utils import count_layers


def layer_counts(
    model: nn.Module,
    *layer_types: type[nn.Module],
) -> dict[type[nn.Module], int]:
    """Count occurrences of each requested layer type in a module."""
    if not layer_types:
        layer_types = (nn.Linear, nn.Sequential, nn.Conv2d)
    return {layer: count_layers(model, layer) for layer in layer_types}


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
        """Default call: split the last layer into heads."""
        model = request.getfixturevalue(model_fixture)
        num_heads = 5

        subensemble_model = subensemble(model, num_heads=num_heads)
        backbone = subensemble_model[0]
        heads = subensemble_model[1]

        counts_original = layer_counts(model)
        counts_backbone = layer_counts(backbone)
        counts_heads = layer_counts(heads)

        assert isinstance(subensemble_model, nn.Module)
        assert isinstance(backbone, nn.Sequential)
        assert isinstance(heads, nn.ModuleList)

        # last layer goes to the heads
        assert counts_heads[nn.Linear] == num_heads
        assert counts_backbone[nn.Linear] == counts_original[nn.Linear] - 1
        assert (counts_original[nn.Linear] + num_heads - 1) == (counts_heads[nn.Linear] + counts_backbone[nn.Linear])

        # structure stays consistent
        assert counts_heads[nn.Sequential] == num_heads
        assert counts_backbone[nn.Sequential] == counts_original[nn.Sequential]

        # conv layers should always stay in the backbone
        assert counts_backbone[nn.Conv2d] == counts_original[nn.Conv2d]
        assert counts_heads[nn.Conv2d] == 0

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
        """Using head_layer=2 should split the last two layers into the head."""
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

        counts_original = layer_counts(model)
        counts_backbone = layer_counts(backbone)
        counts_heads = layer_counts(heads)

        assert isinstance(subensemble_model, nn.Module)
        assert isinstance(backbone, nn.Sequential)
        assert isinstance(heads, nn.ModuleList)

        # Distinguish: both last layers are Linear vs. only the last is Linear.
        if isinstance(model[-2], nn.Linear):
            # Pure linear stack: last two layers are Linear.
            assert counts_heads[nn.Linear] == num_heads * head_layer
            assert counts_backbone[nn.Linear] == counts_original[nn.Linear] - head_layer
            assert counts_original[nn.Linear] == head_layer + counts_backbone[nn.Linear]
        else:
            # Mixed conv/flatten/dropout/relu + final Linear: only last Linear goes to heads.
            assert counts_heads[nn.Linear] == num_heads
            assert counts_backbone[nn.Linear] == counts_original[nn.Linear] - 1

        assert counts_heads[nn.Sequential] == num_heads
        assert counts_backbone[nn.Sequential] == counts_original[nn.Sequential]
        assert counts_backbone[nn.Conv2d] == counts_original[nn.Conv2d]
        assert counts_heads[nn.Conv2d] == 0

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
        """Using an explicit head model keeps base intact and clones the head."""
        model = request.getfixturevalue(model_fixture)
        num_heads = 5

        subensemble_model = subensemble(
            base=model,
            num_heads=num_heads,
            head=model,
        )
        backbone = subensemble_model[0]
        heads = subensemble_model[1]

        counts_original = layer_counts(model)
        counts_backbone = layer_counts(backbone)
        counts_heads = layer_counts(heads)

        assert isinstance(subensemble_model, nn.Module)
        assert isinstance(backbone, nn.Sequential)
        assert isinstance(heads, nn.ModuleList)

        # backbone == original model
        assert counts_backbone[nn.Linear] == counts_original[nn.Linear]
        assert counts_backbone[nn.Conv2d] == counts_original[nn.Conv2d]
        assert counts_backbone[nn.Sequential] == counts_original[nn.Sequential]

        # heads = num_heads clones of the original model
        assert counts_heads[nn.Linear] == counts_original[nn.Linear] * num_heads
        assert counts_heads[nn.Conv2d] == counts_original[nn.Conv2d] * num_heads
        assert counts_heads[nn.Sequential] == num_heads

        # total number of linear layers matches expectation
        assert counts_heads[nn.Linear] + counts_backbone[nn.Linear] == counts_original[nn.Linear] * (num_heads + 1)


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

    def test_zero_heads_returns_empty_heads(
        self,
        torch_model_small_2d_2d: nn.Module,
    ) -> None:
        """num_heads=0 should produce an empty head list and a valid backbone."""
        num_heads = 0

        subensemble_model = subensemble(
            torch_model_small_2d_2d,
            num_heads=num_heads,
        )
        backbone, heads = subensemble_model
        counts_original = layer_counts(torch_model_small_2d_2d)
        counts_backbone = layer_counts(backbone)
        counts_heads = layer_counts(heads)

        # heads should be an empty ModuleList
        assert isinstance(heads, nn.ModuleList)
        assert len(heads) == num_heads

        # backbone should still contain all but the default head layer
        assert counts_backbone[nn.Linear] == counts_original[nn.Linear] - 1
        assert counts_backbone[nn.Conv2d] == counts_original[nn.Conv2d]
        assert counts_backbone[nn.Sequential] == counts_original[nn.Sequential]
        assert counts_heads[nn.Linear] == 0
        assert counts_heads[nn.Conv2d] == 0
        assert counts_heads[nn.Sequential] == 0

    def test_head_layer_zero_raises_value_error(
        self,
        torch_model_small_2d_2d: nn.Module,
    ) -> None:
        """head_layer <= 0 should raise a ValueError exception."""
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

    def test_large_head_layer_uses_all_layers_as_head(
        self,
        torch_model_small_2d_2d: nn.Module,
    ) -> None:
        """A very large head_layer should move all layers into the head."""
        num_heads = 2
        head_layer = count_layers(torch_model_small_2d_2d, nn.Linear)

        subensemble_model = subensemble(
            torch_model_small_2d_2d,
            num_heads=num_heads,
            head_layer=head_layer,
        )
        backbone, heads = subensemble_model
        counts_original = layer_counts(torch_model_small_2d_2d)
        counts_backbone = layer_counts(backbone)
        counts_heads = layer_counts(heads)

        # backbone becomes empty, since all layers are taken into the head.
        assert counts_backbone[nn.Linear] == 0
        assert counts_backbone[nn.Conv2d] == 0
        assert counts_backbone[nn.Sequential] == counts_original[nn.Sequential]

        # heads contain num_heads copies of the full original model.
        assert isinstance(heads, nn.ModuleList)
        assert len(heads) == num_heads
        assert counts_heads[nn.Linear] == head_layer * num_heads
        assert counts_heads[nn.Conv2d] == 0
        assert counts_heads[nn.Sequential] == num_heads
