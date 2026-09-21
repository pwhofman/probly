"""Common functions for the reset traverser."""

from __future__ import annotations

from collections.abc import Callable

from pytraverse import flexdispatch_traverser

reset_traverser = flexdispatch_traverser[object](name="reset_traverser")


# Postorder traversal may visit bare callables before a backend's module handlers
# are loaded. Backend-specific module handlers take precedence over this fallback.
@reset_traverser.register(cls=Callable)
def _reset_callable(obj: Callable) -> Callable:
    """Leave bare callables, such as activation functions, untouched."""
    return obj


@reset_traverser.register
def _(obj: object) -> object:
    msg = f"resetting parameters of {type(obj)} models is not supported yet."
    raise NotImplementedError(msg)
