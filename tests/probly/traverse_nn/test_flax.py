from __future__ import annotations

import pytest

from probly.traverse_nn import find_layer, find_layers

flax = pytest.importorskip("flax")

from flax import nnx  # noqa: E402


def test_find_layers_returns_all_matches_in_order(flax_dropout_model: nnx.Sequential) -> None:
    found = find_layers(flax_dropout_model, nnx.Linear)

    assert found == [flax_dropout_model.layers[0], flax_dropout_model.layers[2]]


def test_find_layers_traverses_custom_modules(flax_custom_model: nnx.Module) -> None:
    found = find_layers(flax_custom_model, nnx.Linear)

    assert [layer.in_features for layer in found] == [10, 20]


def test_find_layers_returns_empty_list_without_match(flax_dropout_model: nnx.Sequential) -> None:
    assert find_layers(flax_dropout_model, nnx.Conv) == []


def test_find_layer_returns_first_match(flax_dropout_model: nnx.Sequential) -> None:
    assert find_layer(flax_dropout_model, nnx.Dropout) is flax_dropout_model.layers[1]


def test_find_layer_raises_without_match(flax_dropout_model: nnx.Sequential) -> None:
    with pytest.raises(ValueError, match="No layer of type"):
        find_layer(flax_dropout_model, nnx.Conv)
