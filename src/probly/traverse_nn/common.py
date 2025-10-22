"""Generic traverser helpers for neural networks."""

from __future__ import annotations

import pytraverse as t

LAYER_COUNT = t.GlobalVariable[int](
    "LAYER_COUNT",
    "The DFS index of the current layer/module.",
    default=0,
)
FLATTEN_SEQUENTIAL = t.StackVariable[bool](
    "FLATTEN_SEQUENTIAL",
    "Whether to flatten sequential modules after making changes.",
    default=True,
)


@t.computed
def is_first_layer(state: t.State) -> bool:
    """Whether the current layer is the first layer."""
    return state[LAYER_COUNT] == 0


layer_count_traverser = t.singledispatch_traverser[object](name="layer_count_traverser")

nn_traverser = t.lazydispatch_traverser[object](name="nn_traverser")


def compose(
    traverser: t.Traverser,
    nn_traverser: t.Traverser = nn_traverser,
    name: str | None = None,
) -> t.Traverser:
    """Compose a custom traverser with neural network traversal functionality.

    This function creates a sequential traverser that combines neural network traversal,
    a custom traverser, and layer counting capabilities in a specific order.

    Args:
        traverser: A custom traverser function to be composed with the NN traverser.
        nn_traverser: The neural network traverser to use. Defaults to the module's
            nn_traverser.
        name: Optional name for the composed traverser.

    Returns:
        A composed sequential traverser that applies NN traversal, custom traversal,
        and layer counting in sequence.
    """
    return t.sequential(nn_traverser, traverser, layer_count_traverser, name=name)
