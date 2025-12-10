"""Common methods for temperature scaling."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lazy_dispatch.singledispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType


@lazydispatch
def _temperature_factory(base: object, device: object) -> type[Any]:
    message = f"No Temperature implementation for base={type(base)}, device={type(device)}"
    raise NotImplementedError(message)

def register_temperature_factory(key: LazyType) -> Callable:
    """Returns a decorator to register a class in the temperature factory."""
    return _temperature_factory.register(key)

class Temperature:
    """Dispatcher, for the different temperature scaling implementations."""

    def __new__(cls, base: object, device: object) -> object:
        """Dispatches to the correct class, creates an instance and returns it."""
        implementation: type[Any] = _temperature_factory(base, device)
        return implementation(base, device)
