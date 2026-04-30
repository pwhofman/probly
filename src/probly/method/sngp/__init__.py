"""SNGP: Spectral-normalized Neural Gaussian Process implementation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import SNGPPredictor, register, sngp, sngp_traverser


## Torch
@sngp_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "SNGPPredictor",
    "register",
    "sngp",
    "sngp_traverser",
]
