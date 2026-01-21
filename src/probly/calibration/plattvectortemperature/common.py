"""Common methods for temperature scaling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lazy_dispatch.singledispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType


@lazydispatch
def _temperature_factory(base: object) -> type[Any]:
    message = f"No Temperature implementation for base={type(base)}"
    raise NotImplementedError(message)


def register_temperature_factory(key: LazyType) -> Callable:
    """Returns a decorator to register a class in the temperature factory."""
    return _temperature_factory.register(key)


def temperature(base: object) -> object:
    """Dispatches to the correct temperature scaling implementation."""
    implementation: type[Any] = _temperature_factory(base)
    return implementation(base)


@lazydispatch
def _affine_factory(base: object, num_classes: int) -> type[Any]:
    message = f"No platt/vector scaling implementation for base={type(base)}"
    raise NotImplementedError(message)


def register_affine_factory(key: LazyType) -> Callable:
    """Returns a decorator to register a class in the affine factory."""
    return _affine_factory.register(key)


def affine(base: object, num_classes: int) -> object:
    """Dispatches to the correct temperature scaling implementation."""
    implementation: type[Any] = _affine_factory(base, num_classes)
    return implementation(base, num_classes)
