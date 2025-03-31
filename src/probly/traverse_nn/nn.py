"""
Generic traverser helpers for neural networks.

"""

from typing import Any

import probly.traverse as t

LAYER_COUNT = t.GlobalVariable[int](
    "LAYER_COUNT", "The DFS index of the current layer/module.", default=0
)


@t.computed
def is_first_layer(s) -> bool:
    "Whether the current layer is the first layer."
    return s[LAYER_COUNT] == 0


layer_count_traverser = t.singledispatch_traverser(name="layer_count_traverser")

nn_traverser = t.singledispatch_traverser(name="nn_traverser")


def compose[T](
    traverser: t.Traverser[T],
    nn_traverser: t.Traverser[T] = nn_traverser,
    name: str | None = None,
) -> t.Traverser[T]:
    return t.sequential(nn_traverser, traverser, layer_count_traverser, name=name)


def traverse[T](
    obj: T,
    traverser: t.Traverser[T],
    nn_traverser: t.Traverser[T] = nn_traverser,
    init: dict[t.Variable, Any] | None = None,
) -> T:
    return t.traverse(obj, compose(traverser, nn_traverser), init=init)
