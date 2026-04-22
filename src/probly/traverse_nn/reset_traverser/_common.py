"""Common functions for the reset traverser."""

from __future__ import annotations

from pytraverse import flexdispatch_traverser

reset_traverser = flexdispatch_traverser[object](name="reset_traverser")


@reset_traverser.register
def _(obj: object) -> object:
    msg = f"resetting parameters of {type(obj)} models is not supported yet."
    raise NotImplementedError(msg)
