"""Module for posterior network implementations."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from . import _common
from ._common import posterior_network_generator


## Torch
@posterior_network_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


posterior_network = _common.posterior_network

__all__ = [
    "posterior_network",
]
