"""Credal net implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import credal_net, credal_net_generator, register


## Torch
@credal_net_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    ...


__all__ = [
    "credal_net",
    "register",
]
