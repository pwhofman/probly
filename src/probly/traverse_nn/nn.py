"""Generic traverser helpers for neural networks."""

from __future__ import annotations

from typing import Any

import probly.traverse as t

LAYER_COUNT = t.GlobalVariable[int](
    "LAYER_COUNT", "The DFS index of the current layer/module.", default=0
)


@t.computed
def is_first_layer(state: t.State) -> bool:
    """Whether the current layer is the first layer."""
    return state[LAYER_COUNT] == 0


layer_count_traverser = t.singledispatch_traverser(name="layer_count_traverser")

nn_traverser = t.singledispatch_traverser(name="nn_traverser")


def compose[T](
    traverser: t.Traverser[T],
    nn_traverser: t.Traverser[T] = nn_traverser,
    name: str | None = None,
) -> t.Traverser[T]:
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


def traverse[T](
    obj: T,
    traverser: t.Traverser[T],
    nn_traverser: t.Traverser[T] = nn_traverser,
    init: dict[t.Variable, Any] | None = None,
) -> T:
    """Traverse a neural network object using a combination of traversers.

    This function applies a composed traverser that combines a user-provided traverser
    with a neural network-specific traverser to walk through and potentially transform
    a neural network object structure.

    Args:
        obj (T): The object to traverse, typically a neural network or related structure.
        traverser (t.Traverser[T]): User-defined traverser function that specifies
            how to process or transform the object during traversal.
        nn_traverser (t.Traverser[T], optional): Neural network-specific traverser.
            Defaults to the module-level nn_traverser.
        init (dict[t.Variable, Any] | None, optional): Initial variable bindings
            or state to use during traversal. Defaults to None.

    Returns:
        T: The traversed (and potentially transformed) object of the same type as input.

    Example:
        >>> model = SomeNeuralNetwork()
        >>> def my_traverser(x): return x * 2
        >>> result = traverse(model, my_traverser)
    """
    return t.traverse(obj, compose(traverser, nn_traverser), init=init)
