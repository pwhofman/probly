from __future__ import annotations

import torch
from torch import nn

from probly.transformation.subensemble import subensemble
from tests.probly.torch_utils import count_layers


def test_subensemble_default(torch_model_small_2d_2d: nn.Module) -> None:
    """Default call splits the last layer into heads."""
    num_heads = 5

    subensemble_model = subensemble(torch_model_small_2d_2d, num_heads=num_heads)

    # Top-level structure: [backbone, heads]
    assert isinstance(subensemble_model, nn.ModuleList)
    assert len(subensemble_model) == 2

    backbone, heads = subensemble_model

    assert isinstance(backbone, nn.Module)
    assert isinstance(heads, nn.ModuleList)
    assert len(heads) == num_heads

    # Original model: 3 x Linear(2, 2)
    # Default head_layer=1:
    #   backbone -> first 2 Linear layers
    #   heads    -> num_heads x 1 Linear layer (wrapped in Sequential)
    assert count_layers(backbone, nn.Linear) == 2
    assert count_layers(backbone, nn.Conv2d) == 0
    assert count_layers(backbone, nn.Sequential) == 1

    assert count_layers(heads, nn.Linear) == num_heads
    assert count_layers(heads, nn.Conv2d) == 0
    assert count_layers(heads, nn.Sequential) == num_heads


def test_subensemble_2_head_layer(torch_model_small_2d_2d: nn.Module) -> None:
    """Using head_layer=2 should split the last two layers into the head."""
    num_heads = 5
    head_layers = 2

    subensemble_model = subensemble(
        torch_model_small_2d_2d,
        num_heads=num_heads,
        head_layer=head_layers,
    )

    backbone, heads = subensemble_model

    # Original: 3 Linear layers
    # head_layer=2:
    #   backbone -> first 1 Linear layer
    #   heads    -> num_heads x 2 Linear layers
    assert count_layers(backbone, nn.Linear) == 1
    assert count_layers(backbone, nn.Conv2d) == 0
    assert count_layers(backbone, nn.Sequential) == 1

    assert len(heads) == num_heads
    assert count_layers(heads, nn.Linear) == head_layers * num_heads
    assert count_layers(heads, nn.Conv2d) == 0
    assert count_layers(heads, nn.Sequential) == num_heads

    # Each head should consist of exactly two Linear layers.
    first_head = heads[0]
    head_layers_modules = list(first_head.children())
    assert len(head_layers_modules) == head_layers
    assert all(isinstance(layer, nn.Linear) for layer in head_layers_modules)


def test_subensemble_with_head_model(torch_model_small_2d_2d: nn.Module) -> None:
    """Using an explicit head model keeps obj intact and clones the head."""
    num_heads = 5

    subensemble_model = subensemble(
        torch_model_small_2d_2d,
        num_heads=num_heads,
        head=torch_model_small_2d_2d,
    )

    backbone, heads = subensemble_model

    # Backbone should correspond to the full original model (3 Linear layers).
    assert isinstance(backbone, nn.Sequential)
    assert len(list(backbone.children())) == len(
        list(torch_model_small_2d_2d.children()),
    )
    assert count_layers(backbone, nn.Linear) == 3
    assert count_layers(backbone, nn.Conv2d) == 0
    assert count_layers(backbone, nn.Sequential) == 1

    # Heads should be num_heads clones of the head model.
    assert isinstance(heads, nn.ModuleList)
    assert len(heads) == num_heads
    assert count_layers(heads, nn.Linear) == 3 * num_heads
    assert count_layers(heads, nn.Conv2d) == 0
    assert count_layers(heads, nn.Sequential) == num_heads

    # Check that at least two heads have independent parameters.
    params0 = next(iter(heads[0].parameters())).detach().clone()
    params1 = next(iter(heads[1].parameters())).detach().clone()
    assert params0.data_ptr() != params1.data_ptr()


def test_subensemble_with_head_model_and_reset_params(
    torch_model_small_2d_2d: nn.Module,
) -> None:
    """reset_params=True with an explicit head should only reset cloned heads."""
    num_heads = 2
    original_params = [p.detach().clone() for p in torch_model_small_2d_2d.parameters()]

    subensemble_model = subensemble(
        torch_model_small_2d_2d,
        num_heads=num_heads,
        head=torch_model_small_2d_2d,
        reset_params=True,
    )

    backbone, heads = subensemble_model

    assert backbone is torch_model_small_2d_2d
    assert len(heads) == num_heads

    params0 = next(iter(heads[0].parameters())).detach().clone()
    params1 = next(iter(heads[1].parameters())).detach().clone()
    assert params0.data_ptr() != params1.data_ptr()
    assert not torch.equal(params0, params1)

    backbone_params = list(backbone.parameters())
    assert all(
        torch.equal(p_before, p_after) for p_before, p_after in zip(original_params, backbone_params, strict=False)
    )


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

        # heads should be an empty ModuleList
        assert isinstance(heads, nn.ModuleList)
        assert len(heads) == 0

        # backbone should still contain all but the default head layer
        assert count_layers(backbone, nn.Linear) == 2
        assert count_layers(backbone, nn.Conv2d) == 0
        assert count_layers(backbone, nn.Sequential) == 1

    def test_head_layer_zero_moves_all_layers_to_head(
        self,
        torch_model_small_2d_2d: nn.Module,
    ) -> None:
        """head_layer=0 should move all layers into the head and leave backbone empty."""
        num_heads = 3
        original_params = [p.detach().clone() for p in torch_model_small_2d_2d.parameters()]

        subensemble_model = subensemble(
            torch_model_small_2d_2d,
            num_heads=num_heads,
            head_layer=0,
        )
        backbone, heads = subensemble_model

        # backbone becomes an empty Sequential
        assert isinstance(backbone, nn.Sequential)
        assert len(list(backbone.children())) == 0
        assert count_layers(backbone, nn.Linear) == 0
        assert count_layers(backbone, nn.Conv2d) == 0
        assert count_layers(backbone, nn.Sequential) == 1

        # heads contain num_heads copies of the full original model.
        assert isinstance(heads, nn.ModuleList)
        assert len(heads) == num_heads
        assert count_layers(heads, nn.Linear) == 3 * num_heads
        assert count_layers(heads, nn.Conv2d) == 0
        assert count_layers(heads, nn.Sequential) == num_heads

        # original model remains unchanged
        assert count_layers(torch_model_small_2d_2d, nn.Linear) == 3
        assert all(
            torch.equal(p_before, p_after)
            for p_before, p_after in zip(
                original_params,
                torch_model_small_2d_2d.parameters(),
                strict=False,
            )
        )

    def test_large_head_layer_uses_all_layers_as_head(
        self,
        torch_model_small_2d_2d: nn.Module,
    ) -> None:
        """A very large head_layer should move all layers into the head."""
        num_heads = 2
        head_layer = 100

        subensemble_model = subensemble(
            torch_model_small_2d_2d,
            num_heads=num_heads,
            head_layer=head_layer,
        )
        backbone, heads = subensemble_model

        # backbone becomes empty, since all layers are taken into the head.
        assert count_layers(backbone, nn.Linear) == 0
        assert count_layers(backbone, nn.Conv2d) == 0
        assert count_layers(backbone, nn.Sequential) == 1

        # heads contain num_heads copies of the full original model.
        assert isinstance(heads, nn.ModuleList)
        assert len(heads) == num_heads
        assert count_layers(heads, nn.Linear) == 3 * num_heads
        assert count_layers(heads, nn.Conv2d) == 0
        assert count_layers(heads, nn.Sequential) == num_heads
