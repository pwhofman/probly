"""DropConnect implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from . import common

dropconnect = common.dropconnect
register = common.register


## Torch
@common.dropconnect_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415
