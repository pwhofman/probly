"""Common methods for BBQ Calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lazy_dispatch.singledispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType


# dispatches based on the type of the base model and device
@lazydispatch
def _bayesian_binning_factory(base: object, device: object) -> type[Any]:
    msg = f"No Bayesian Binning into Quantiles implementation for base={type(base)}, device={type(device)}"
    raise NotImplementedError(msg)


# decorator to register a histogrambinnig class for a base type
def register_bayesian_binning_factory(key: LazyType) -> Callable:
    """Decorator to register a Bayesian Binning into Quantiles class for a base type."""
    return _bayesian_binning_factory.register(key)
