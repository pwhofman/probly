"""Test for torch classification models."""

from __future__ import annotations

from torch import nn

from probly.layers.torch import DropConnectLinear
from probly.transformation.dropconnect.common import (
    dropconnect,
)
from tests.probly.torch_utils import count_layers


def test_evidential_classification_appends_softplus_on_linear(torch_model_small_2d_2d: nn.Sequential) -> None:
    model = dropconnect(torch_model_small_2d_2d)

    # count number of nn.Linear layers in original model
    count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
    # count number of dropconnect layers in original model
    count_dropconnect_original = count_layers(torch_model_small_2d_2d, DropConnectLinear)
    # count number of nn.Sequential layers in original model
    count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

    # count number of nn.Linear layers in modified model
    count_linear_modified = count_layers(model, nn.Linear)
    # count number of dropconnect layers in modified model
    count_dropconnect_modified = count_layers(model, DropConnectLinear)
    # count number of nn.Sequential layers in modified model
    count_sequential_modified = count_layers(model, nn.Sequential)

    linear_diff = count_linear_original - count_linear_modified
    dropconnect_diff = count_dropconnect_modified - count_dropconnect_original
    # check that the model is not modified except for the softplus layer at the end of the new sequence layer
    assert model is not None
    assert isinstance(model, type(torch_model_small_2d_2d))
    assert linear_diff == dropconnect_diff
    assert count_sequential_original == count_sequential_modified


def test_evidential_classification_appends_softplus_on_conv(torch_conv_linear_model: nn.Sequential) -> None:
    model = dropconnect(torch_conv_linear_model)

    # count number of nn.Linear layers in original model
    count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
    # count number of dropconnect layers in original model
    count_dropconnect_original = count_layers(torch_conv_linear_model, DropConnectLinear)
    # count number of nn.Sequential layers in original model
    count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
    # count number of nn.Conv2d layers in original model
    count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

    # count number of nn.Linear layers in modified model
    count_linear_modified = count_layers(model, nn.Linear)
    # count number of dropconnect layers in modified model
    count_dropconnect_modified = count_layers(model, DropConnectLinear)
    # count number of nn.Sequential layers in modified model
    count_sequential_modified = count_layers(model, nn.Sequential)
    # count number of nn.Conv2d layers in modified model
    count_conv_modified = count_layers(model, nn.Conv2d)

    linear_diff = count_linear_original - count_linear_modified
    dropconnect_diff = count_dropconnect_modified - count_dropconnect_original
    # check that the model is not modified except for the softplus layer at the end of the new sequence layer
    assert model is not None
    assert isinstance(model, type(torch_conv_linear_model))
    assert linear_diff == dropconnect_diff
    assert count_sequential_original == count_sequential_modified
    assert count_conv_original == count_conv_modified


def test_custom_network(torch_custom_model: nn.Module) -> None:
    """Tests the custom model modification with added dropconnect layers."""
    model = dropconnect(torch_custom_model)

    # check if model type is correct
    assert type(model) is type(torch_custom_model)
