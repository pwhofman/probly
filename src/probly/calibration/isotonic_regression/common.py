"""Common methods for isotonic regression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lazy_dispatch.singledispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType


@lazydispatch
def _isotonic_factory(base_model: object, use_logits: bool) -> type[Any]:
    message = f"No Isotonic Regression implementation for base={type(base_model)}, use_logits={use_logits}."
    raise NotImplementedError(message)


def register_isotonic_factory(key: LazyType) -> Callable:
    """Returns a decorator to register a class in the isotonic factory."""
    return _isotonic_factory.register(key)


def isotonic_regression(base: object, use_logits: bool) -> object:
    """Dispatches different implementations for isotonic regression."""
    implementation: type[Any] = _isotonic_factory(base, use_logits)
    return implementation(base, use_logits)
