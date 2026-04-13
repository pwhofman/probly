"""Credal net implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import CredalNetPredictor, credal_net, credal_net_traverser


## Torch
@credal_net_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "CredalNetPredictor",
    "credal_net",
]
