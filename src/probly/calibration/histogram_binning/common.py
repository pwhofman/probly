from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING, Any

from lazy_dispatch.singledispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType


# dispatches based on the type of the base model and device
@lazydispatch
def _histogram_factory(base: object, device: object) -> type[Any]:
    msg = f"No HistogramBinning implementation for base={type(base)}, device={type(device)}"
    raise NotImplementedError(msg)


# decorator to register a histogrambinnig class for a base type
def register_histogram_factory(key: LazyType) -> Callable:
    """Decorator to register a HistogramBinning class for a base type."""
    return _histogram_factory.register(key)
