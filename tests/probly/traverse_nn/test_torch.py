from __future__ import annotations

import pytest

from probly.traverse_nn import find_layer, find_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


def test_find_layers_returns_all_matches_in_order(torch_dropout_model: nn.Sequential) -> None:
    found = find_layers(torch_dropout_model, nn.Linear)

    assert found == [torch_dropout_model[0], torch_dropout_model[3]]


def test_find_layers_with_type_tuple(torch_dropout_model: nn.Sequential) -> None:
    found = find_layers(torch_dropout_model, (nn.ReLU, nn.Dropout))

    assert found == [torch_dropout_model[1], torch_dropout_model[2]]


def test_find_layers_traverses_custom_modules(torch_custom_model: nn.Module) -> None:
    found = find_layers(torch_custom_model, nn.Linear)

    assert found == [torch_custom_model.linear1, torch_custom_model.linear2]


def test_find_layers_returns_empty_list_without_match(torch_dropout_model: nn.Sequential) -> None:
    assert find_layers(torch_dropout_model, nn.Conv2d) == []


def test_find_layers_does_not_clone(torch_dropout_model: nn.Sequential) -> None:
    found = find_layers(torch_dropout_model, nn.Dropout)

    assert found[0] is torch_dropout_model[2]


def test_find_layer_returns_first_match(torch_dropout_model: nn.Sequential) -> None:
    assert find_layer(torch_dropout_model, nn.Linear) is torch_dropout_model[0]


def test_find_layer_raises_without_match(torch_dropout_model: nn.Sequential) -> None:
    with pytest.raises(ValueError, match="No layer of type"):
        find_layer(torch_dropout_model, nn.Conv2d)
