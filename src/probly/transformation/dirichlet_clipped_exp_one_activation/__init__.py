"""Dirichlet clipped-exp + 1 activation transformation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE
from probly.transformation.dirichlet_clipped_exp_one_activation import _common
from probly.transformation.dirichlet_clipped_exp_one_activation._common import (
    DirichletClippedExpOneActivationPredictor,
)

dirichlet_clipped_exp_one_activation = _common.dirichlet_clipped_exp_one_activation
register = _common.register


## Torch
@_common.dirichlet_clipped_exp_one_activation_appender.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "DirichletClippedExpOneActivationPredictor",
    "dirichlet_clipped_exp_one_activation",
    "register",
]
