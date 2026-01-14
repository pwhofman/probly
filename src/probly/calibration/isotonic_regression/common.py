"""Common methods for isotonic regression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lazy_dispatch.singledispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType


@lazydispatch
def _isotonic_factory(base: object, device: object) -> type[Any]:
    message = f"No Isotonic Regression implementation for base={type(base)}, device={type(device)}"
    raise NotImplementedError(message)


def register_isotonic_factory(key: LazyType) -> Callable:
    """Returns a decorator to register a class in the isotonic factory."""
    return _isotonic_factory.register(key)


def isotonic_regression(base: object, device: object) -> object:
    """Dispatches different implementations for isotonic regression."""
    return _isotonic_factory(base, device)
