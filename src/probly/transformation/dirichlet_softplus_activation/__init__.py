"""Dirichlet softplus-activation transformation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE
from probly.transformation.dirichlet_softplus_activation import _common
from probly.transformation.dirichlet_softplus_activation._common import DirichletSoftplusActivationPredictor

dirichlet_softplus_activation = _common.dirichlet_softplus_activation
register = _common.register


## Torch
@_common.dirichlet_softplus_activation_appender.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["DirichletSoftplusActivationPredictor", "dirichlet_softplus_activation", "register"]
