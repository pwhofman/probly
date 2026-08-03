"""Train functionality for variational Bayesian last layer (VBLL) models."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import vbll_loss


## Torch
@vbll_loss.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "vbll_loss",
]
