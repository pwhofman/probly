"""Generic utils for traverse_nn."""

from __future__ import annotations

from typing import Any

from probly.lazy_types import TORCH_MODULE
import pytraverse as t
from pytraverse.generic import CLONE as GENERIC_CLONE

from ._common import compose as nn_compose

LAST_OUTPUT_DIM = t.GlobalVariable[int]("LAST_OUTPUT_DIM", default=-1)
"""Global variable tracking the output dim of the most recently visited layer during traversal."""


output_dim_traverser = t.flexdispatch_traverser[object](name="output_dim_traverser")
"""Dispatches on layer type. Registered handlers update :data:`LAST_OUTPUT_DIM` and return the
layer unchanged. Backend-specific handlers (e.g., for torch ``nn.Linear``/``nn.Conv2d``) are
registered in the corresponding backend util module."""


@output_dim_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch_utils  # noqa: F401, PLC0415


def get_output_dim(model: object) -> int:
    """Return the output feature dimension of a model.

    Walks ``model`` using the neural-network traverser and returns the output feature
    dimension of the last weight-bearing layer visited in forward DFS order. For typical
    classifier-style encoders (e.g., torchvision ResNet) this corresponds to the final
    ``Linear`` layer's ``out_features`` or the final ``Conv`` layer's ``out_channels``.

    The walk does not mutate or deep-copy the input model. Backend-specific handlers must
    be registered on :data:`output_dim_traverser` for each layer type that should contribute
    an output dim (for torch, this is done in :mod:`probly.utils.torch`).

    Args:
        model: The model whose output dim to infer. Must be of a type for which handlers
            have been registered (currently torch ``nn.Module``).

    Returns:
        The number of output features of the model's last weight-bearing layer.

    Raises:
        ValueError: If no weight-bearing layer was visited during traversal (e.g., the
            model contains no ``Linear`` or ``Conv`` layer, or no handlers are registered
            for the model's backend).
    """
    _, state = t.traverse_with_state(
        model,
        nn_compose(output_dim_traverser),
        init={GENERIC_CLONE: False, LAST_OUTPUT_DIM: -1},
    )
    dim = state[LAST_OUTPUT_DIM]
    if dim < 0:
        msg = (
            f"Could not infer output dim for {type(model).__name__}: no weight-bearing "
            "layer was visited during traversal. Is a handler registered for this backend?"
        )
        raise ValueError(msg)
    return dim


LAYER_TYPES = t.GlobalVariable[tuple[type, ...]]("LAYER_TYPES", "The layer types to collect during traversal.")
FOUND_LAYERS = t.GlobalVariable[list[Any]]("FOUND_LAYERS", "The matching layers collected during traversal.")


@t.traverser
def layer_finder(obj: object, state: t.State[Any]) -> t.TraverserResult[object]:
    """Collect visited objects that are instances of :data:`LAYER_TYPES` into :data:`FOUND_LAYERS`."""
    if isinstance(obj, state[LAYER_TYPES]):
        state[FOUND_LAYERS].append(obj)
    return obj, state


def find_layers(model: object, layer_types: type | tuple[type, ...]) -> list[Any]:
    """Return all layers of the given type(s) contained in a model.

    Walks ``model`` using the neural-network traverser and collects every visited
    layer that is an instance of ``layer_types``, in forward DFS order. The check
    is a plain ``isinstance``, so no per-backend handlers are needed; any backend
    supported by :data:`~probly.traverse_nn.nn_traverser` (currently torch and
    flax NNX) works out of the box. The walk does not mutate or deep-copy the model.

    Args:
        model: The model to search.
        layer_types: A layer type or tuple of layer types to match.

    Returns:
        All matching layers in traversal order; empty if none match.
    """
    found: list[Any] = []
    t.traverse(
        model,
        nn_compose(layer_finder),
        init={GENERIC_CLONE: False, LAYER_TYPES: layer_types, FOUND_LAYERS: found},
    )
    return found


def find_layer(model: object, layer_types: type | tuple[type, ...]) -> Any:  # noqa: ANN401, the layer type depends on the requested types
    """Return the first layer of the given type(s) contained in a model.

    Convenience wrapper around :func:`find_layers` for the common case of a model
    holding a single layer of interest (e.g. a swapped-in last layer).

    Args:
        model: The model to search.
        layer_types: A layer type or tuple of layer types to match.

    Returns:
        The first matching layer in forward DFS order.

    Raises:
        ValueError: If the model contains no layer of the given type(s).
    """
    layers = find_layers(model, layer_types)
    if not layers:
        msg = f"No layer of type(s) {layer_types!r} found in {type(model).__name__}."
        raise ValueError(msg)
    return layers[0]
