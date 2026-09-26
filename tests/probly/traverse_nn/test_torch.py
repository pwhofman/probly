from __future__ import annotations

import pytest

from probly.traverse_nn import find_layer, find_layers, is_first_layer, nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, State, traverse, traverser

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


def _modules_with_parameters_seen_as_first_layer(model: nn.Module, *, reverse: bool) -> list[nn.Module]:
    """Walk a model and return the modules with parameters for which ``is_first_layer`` holds."""
    first: list[nn.Module] = []

    @traverser
    def record(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
        if is_first_layer(state) and any(True for _ in obj.parameters(recurse=False)):
            first.append(obj)
        return obj, state

    traverse(
        model,
        nn_compose(record),
        init={CLONE: False, TRAVERSE_REVERSED: reverse},
    )
    return first


class TestFirstLayerDetection:
    """``is_first_layer`` refers to the first module with parameters; parameter-free modules are ignored."""

    def test_leading_parameter_free_modules_are_ignored(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Identity(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))

        assert _modules_with_parameters_seen_as_first_layer(model, reverse=False) == [model[2]]

    def test_trailing_parameter_free_modules_are_ignored_in_reverse(self) -> None:
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3), nn.Softmax(dim=-1), nn.Dropout())

        assert _modules_with_parameters_seen_as_first_layer(model, reverse=True) == [model[2]]

    def test_parameter_free_modules_registered_after_the_output_layer_are_ignored(
        self, torch_trailing_activation_model: nn.Module
    ) -> None:
        model = torch_trailing_activation_model

        assert _modules_with_parameters_seen_as_first_layer(model, reverse=False) == [model.fc1]
        assert _modules_with_parameters_seen_as_first_layer(model, reverse=True) == [model.fc3]

    def test_first_layer_can_be_a_layer_of_any_type_with_parameters(self) -> None:
        """A convolution in front of the linear head is the first layer, so the head is not."""
        model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8, 3))

        assert _modules_with_parameters_seen_as_first_layer(model, reverse=False) == [model[0]]
