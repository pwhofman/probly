from __future__ import annotations

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
