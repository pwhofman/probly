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
def _platt_factory(base: object) -> type[Any]:
    message = f"No platt scaling implementation for base={type(base)}"
    raise NotImplementedError(message)


def register_platt_factory(key: LazyType) -> Callable:
    """Returns a decorator to register a class in the platt factory."""
    return _platt_factory.register(key)


def platt(base: object) -> object:
    """Dispatches to the correct platt scaling implementation."""
    implementation: type[Any] = _platt_factory(base)
    return implementation(base)


@lazydispatch
def _vector_factory(base: object, num_classes: int) -> type[Any]:
    message = f"No vector scaling implementation for base={type(base)}"
    raise NotImplementedError(message)


def register_vector_factory(key: LazyType) -> Callable:
    """Returns a decorator to register a class in the vector factory."""
    return _vector_factory.register(key)


def vector(base: object, num_classes: int) -> object:
    """Dispatches to the correct vector scaling implementation."""
    implementation: type[Any] = _vector_factory(base, num_classes)
    return implementation(base, num_classes)
