from __future__ import annotations

from torch import nn

from probly.transformation.subensemble import subensemble
from tests.probly.torch_utils import count_layers


def test_subensemble_returns_expected_structure(
    torch_model_small_2d_2d: nn.Module,
) -> None:
    """The subensemble should return [backbone, heads] as an nn.ModuleList."""
    num_heads = 4

    model = subensemble(torch_model_small_2d_2d, num_heads)

    assert isinstance(model, nn.ModuleList)
    assert len(model) == 2

    backbone, heads = model

    assert isinstance(backbone, nn.Module)
    assert isinstance(heads, nn.ModuleList)
    assert len(heads) == num_heads


def test_subensemble_creates_expected_number_of_linear_heads(
    torch_model_small_2d_2d: nn.Module,
) -> None:
    """The subensemble should contain exactly num_heads Linear heads."""
    num_heads = 5

    model = subensemble(torch_model_small_2d_2d, num_heads)
    _, heads = model

    count_linear_heads = count_layers(heads, nn.Linear)
    assert count_linear_heads == num_heads


def test_subensemble_splits_backbone_and_head_layers(
    torch_model_small_2d_2d: nn.Module,
) -> None:
    """The default head_layer should split backbone and head correctly."""
    num_heads = 3

    model = subensemble(torch_model_small_2d_2d, num_heads)
    backbone, heads = model

    # original model has 3 Linear layers; backbone should keep first 2
    backbone_layers = list(backbone.children())
    assert len(backbone_layers) == 2
    assert all(isinstance(layer, nn.Linear) for layer in backbone_layers)

    first_head = heads[0]
    head_layers = list(first_head.children())
    assert len(head_layers) == 1
    assert isinstance(head_layers[0], nn.Linear)


def test_subensemble_heads_are_independent(
    torch_model_small_2d_2d: nn.Module,
) -> None:
    """Each head should be an independent module with its own parameters."""
    num_heads = 2

    model = subensemble(torch_model_small_2d_2d, num_heads)
    _, heads = model

    params0 = next(iter(heads[0].parameters())).detach().clone()
    params1 = next(iter(heads[1].parameters())).detach().clone()

    # parameters should not be the same object in memory
    assert params0.data_ptr() != params1.data_ptr()
