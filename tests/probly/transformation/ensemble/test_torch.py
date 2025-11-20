"""Test for torch ensemble models."""

from __future__ import annotations

from torch import nn

from probly.transformation import ensemble
from probly.transformation.ensemble.torch import generate_torch_ensemble
from tests.probly.torch_utils import count_layers


def test_linear_network_with_first_linear(torch_model_small_2d_2d: nn.Sequential) -> None:
    num_members = 5
    model = ensemble(torch_model_small_2d_2d, num_members)

    # count
    count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
    count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)
    count_linear_modified = count_layers(model, nn.Linear)
    count_sequential_modified = count_layers(model, nn.Sequential)

    # check that the model is not modified except for the dropout layer
    assert model is not None
    assert (count_linear_original * num_members) == count_linear_modified
    assert (count_sequential_original * num_members) == count_sequential_modified


def test_convolutional_network(torch_conv_linear_model: nn.Sequential) -> None:
    num_members = 5
    model = ensemble(torch_conv_linear_model, num_members)

    # count
    count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
    count_linear_modified = count_layers(model, nn.Linear)
    count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
    count_sequential_modified = count_layers(model, nn.Sequential)
    count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)
    count_conv_modified = count_layers(model, nn.Conv2d)

    # check that the model is not modified except for the dropout layer
    assert model is not None
    assert (count_linear_original * num_members) == count_linear_modified
    assert (count_sequential_original * num_members) == count_sequential_modified
    assert (count_conv_original * num_members) == count_conv_modified


class DummyModel(nn.Module):
    """Dummy Model for testing parameter reset."""

    def __init__(self) -> None:
        """__init__."""
        super().__init__()
        self.count = 0

    def reset_parameters(self) -> None:
        self.count += 1


def test_reset_copys() -> None:
    base = DummyModel()
    num_members = 3

    genmod = generate_torch_ensemble(base, num_members)

    for members in genmod:
        assert isinstance(members, nn.Module)
        assert members.count == 1
