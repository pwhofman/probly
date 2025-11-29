"""Test for torch classification models."""

from __future__ import annotations

from torch import nn

from probly.transformation.evidential.classification.common import (
    evidential_classification,
)
from tests.probly.torch_utils import count_layers


def test_evidential_classification_appends_softplus_on_linear(torch_model_small_2d_2d: nn.Sequential) -> None:
    model = evidential_classification(torch_model_small_2d_2d)

    # count number of nn.Linear layers in original model
    count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
    # count number of softplus layers in original model
    count_softplus_original = count_layers(torch_model_small_2d_2d, nn.Softplus)
    # count number of nn.Sequential layers in original model
    count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

    # count number of nn.Linear layers in modified model
    count_linear_modified = count_layers(model, nn.Linear)
    # count number of softplus layers in modified model
    count_softplus_modified = count_layers(model, nn.Softplus)
    # count number of nn.Sequential layers in modified model
    count_sequential_modified = count_layers(model, nn.Sequential)

    # check that the model is not modified except for the softplus layer at the end of the new sequence layer
    assert model is not None
    assert isinstance(model, type(torch_model_small_2d_2d))
    assert count_linear_original == count_linear_modified
    assert count_softplus_original == (count_softplus_modified - 1)
    assert count_sequential_original == (count_sequential_modified - 1)


def test_evidential_classification_appends_softplus_on_conv(torch_conv_linear_model: nn.Sequential) -> None:
    model = evidential_classification(torch_conv_linear_model)

    # count number of nn.Linear layers in original model
    count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
    # count number of softplus layers in original model
    count_softplus_original = count_layers(torch_conv_linear_model, nn.Softplus)
    # count number of nn.Sequential layers in original model
    count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
    # count number of nn.Conv2d layers in original model
    count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

    # count number of nn.Linear layers in modified model
    count_linear_modified = count_layers(model, nn.Linear)
    # count number of softplus layers in modified model
    count_softplus_modified = count_layers(model, nn.Softplus)
    # count number of nn.Sequential layers in modified model
    count_sequential_modified = count_layers(model, nn.Sequential)
    # count number of nn.Conv2d layers in modified model
    count_conv_modified = count_layers(model, nn.Conv2d)

    # check that the model is not modified except for the softplus layer at the end of the new sequence layer
    assert model is not None
    assert isinstance(model, type(torch_conv_linear_model))
    assert count_linear_original == count_linear_modified
    assert count_softplus_original == (count_softplus_modified - 1)
    assert count_sequential_original == (count_sequential_modified - 1)
    assert count_conv_original == count_conv_modified
