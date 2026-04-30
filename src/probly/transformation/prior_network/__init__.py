"""Module for Prior Network implementation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE
from probly.transformation.prior_network import _common
from probly.transformation.prior_network._common import PriorNetworkPredictor

prior_network = _common.prior_network
register = _common.register


## Torch
@_common.prior_network_appender.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["PriorNetworkPredictor", "prior_network", "register"]
