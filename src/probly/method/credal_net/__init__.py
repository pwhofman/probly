"""Credal net implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from . import common

credal_net = common.credal_net
register = common.register


## Torch
@common.credal_net_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415
