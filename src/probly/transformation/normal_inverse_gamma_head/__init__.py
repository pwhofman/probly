"""Normal-inverse-gamma head transformation."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE
from probly.transformation.normal_inverse_gamma_head import _common

normal_inverse_gamma_head = _common.normal_inverse_gamma_head
register = _common.register


## Torch
@_common.normal_inverse_gamma_head_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@_common.normal_inverse_gamma_head_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = ["normal_inverse_gamma_head", "register"]
