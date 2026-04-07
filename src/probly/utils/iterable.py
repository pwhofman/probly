"""Utility functions for working with iterables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def first_element[T](elements: Iterable[T], *_args: Any, **_kwargs: Any) -> T:  # noqa: ANN401
    """Get the first element from an iterable.

    Args:
        elements: The elements to create the sample from.
        args: Additional arguments (ignored).
        kwargs: Additional keyword arguments (ignored).

    Returns:
        The first element.
    """
    try:
        return elements[0]  # ty:ignore[not-subscriptable]
    except Exception:  # noqa: BLE001
        return next(iter(elements))
