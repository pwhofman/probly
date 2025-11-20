"""Test for torch classification models."""

from __future__ import annotations

from torch import nn

from probly.layers.torch import NormalInverseGammaLinear
from probly.transformation.evidential.regression.common import (
    evidential_regression,
)
from tests.probly.torch_utils import count_layers


def test_evidential_classification_nig_linear(torch_model_small_2d_2d: nn.Sequential) -> None:
    model = evidential_regression(torch_model_small_2d_2d)

    # count number of nn.Linear layers in original model
    count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
    # count number of nig layers in original model
    count_nig_original = count_layers(torch_model_small_2d_2d, NormalInverseGammaLinear)
    # count number of nn.Sequential layers in original model
    count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

    # count number of nn.Linear layers in modified model
    count_linear_modified = count_layers(model, nn.Linear)
    # count number of nig layers in modified model
    count_nig_modified = count_layers(model, NormalInverseGammaLinear)
    # count number of nn.Sequential layers in modified model
    count_sequential_modified = count_layers(model, nn.Sequential)

    assert model is not None
    assert isinstance(model, type(torch_model_small_2d_2d))
    assert count_nig_original == (count_nig_modified - 1)
    assert count_linear_original == (count_linear_modified + 1)
    assert count_nig_original == (count_nig_modified - 1)
    assert count_sequential_original == count_sequential_modified

    for i in range(len(torch_model_small_2d_2d)):
        if isinstance(torch_model_small_2d_2d[i], nn.Linear):
            last_linear = model[i]
    if last_linear is not None:
        assert isinstance(model[i], NormalInverseGammaLinear)

    for i in range(len(model)):
        if not isinstance(model[i], NormalInverseGammaLinear):
            assert type(model[i]) is type(torch_model_small_2d_2d[i])


def test_evidential_classification_nig_conv(torch_conv_linear_model: nn.Sequential) -> None:
    model = evidential_regression(torch_conv_linear_model)

    # count number of nn.Conv2d layers in original model
    count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
    # count number of nig layers in original model
    count_nig_original = count_layers(torch_conv_linear_model, NormalInverseGammaLinear)
    # count number of nn.Sequential layers in original model
    count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
    # count number of nn.Conv2d layers in original model
    count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

    # count number of nn.Conv2d layers in modified model
    count_linear_modified = count_layers(model, nn.Linear)
    # count number of nig layers in modified model
    count_nig_modified = count_layers(model, NormalInverseGammaLinear)
    # count number of nn.Sequential layers in modified model
    count_sequential_modified = count_layers(model, nn.Sequential)
    # count number of nn.Conv2d layers in modified model
    count_conv_modified = count_layers(model, nn.Conv2d)

    assert model is not None
    assert isinstance(model, type(torch_conv_linear_model))
    assert count_nig_original == (count_nig_modified - 1)
    assert count_linear_original == (count_linear_modified + 1)
    assert count_nig_original == (count_nig_modified - 1)
    assert count_sequential_original == count_sequential_modified
    assert count_conv_original == count_conv_modified

    for i in range(len(model)):
        if not isinstance(model[i], NormalInverseGammaLinear):
            assert type(model[i]) is type(torch_conv_linear_model[i])

    for i in range(len(torch_conv_linear_model)):
        if isinstance(torch_conv_linear_model[i], nn.Linear):
            last_linear = model[i]
    if last_linear is not None:
        assert isinstance(model[i], NormalInverseGammaLinear)


def test_custom_network(torch_custom_model: nn.Module) -> None:
    """Tests the custom model modification with added dropout layers."""
    model = evidential_regression(torch_custom_model)

    # check if model type is correct
    assert type(torch_custom_model) is type(model)
