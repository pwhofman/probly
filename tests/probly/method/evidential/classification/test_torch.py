"""Test for torch classification models."""

from __future__ import annotations

import torch
from torch import nn

from probly.method.evidential.classification import (
    evidential_classification,
)
from probly.transformation.dirichlet_clipped_exp_one_activation.torch import _AddOne, _ClippedExp
from tests.probly.torch_utils import count_layers


def test_evidential_classification_appends_clipped_exp_plus_one_on_linear(
    torch_model_small_2d_2d: nn.Sequential,
) -> None:
    model = evidential_classification(torch_model_small_2d_2d)

    count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
    count_clipped_exp_original = count_layers(torch_model_small_2d_2d, _ClippedExp)
    count_add_one_original = count_layers(torch_model_small_2d_2d, _AddOne)
    count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

    count_linear_modified = count_layers(model, nn.Linear)
    count_clipped_exp_modified = count_layers(model, _ClippedExp)
    count_add_one_modified = count_layers(model, _AddOne)
    count_sequential_modified = count_layers(model, nn.Sequential)

    assert model is not None
    assert isinstance(model, type(torch_model_small_2d_2d))
    assert count_linear_original == count_linear_modified
    assert count_clipped_exp_original == (count_clipped_exp_modified - 1)
    assert count_add_one_original == (count_add_one_modified - 1)
    assert count_sequential_original == (count_sequential_modified - 1)


def test_evidential_classification_appends_clipped_exp_plus_one_on_conv(
    torch_conv_linear_model: nn.Sequential,
) -> None:
    model = evidential_classification(torch_conv_linear_model)

    count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
    count_clipped_exp_original = count_layers(torch_conv_linear_model, _ClippedExp)
    count_add_one_original = count_layers(torch_conv_linear_model, _AddOne)
    count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
    count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

    count_linear_modified = count_layers(model, nn.Linear)
    count_clipped_exp_modified = count_layers(model, _ClippedExp)
    count_add_one_modified = count_layers(model, _AddOne)
    count_sequential_modified = count_layers(model, nn.Sequential)
    count_conv_modified = count_layers(model, nn.Conv2d)

    assert model is not None
    assert isinstance(model, type(torch_conv_linear_model))
    assert count_linear_original == count_linear_modified
    assert count_clipped_exp_original == (count_clipped_exp_modified - 1)
    assert count_add_one_original == (count_add_one_modified - 1)
    assert count_sequential_original == (count_sequential_modified - 1)
    assert count_conv_original == count_conv_modified


def test_evidential_classification_alpha_is_at_least_one(
    torch_model_small_2d_2d: nn.Sequential,
) -> None:
    """The clipped-exp + 1 parameterization must yield ``alpha >= 1`` everywhere."""
    model = evidential_classification(torch_model_small_2d_2d)
    alpha = model(torch.randn(8, 2))
    assert torch.all(alpha >= 1.0)
