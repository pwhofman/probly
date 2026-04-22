"""Generic utils for traverse_nn."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE
from probly.traverse_nn import nn_compose
import pytraverse as t
from pytraverse.generic import CLONE as GENERIC_CLONE

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
