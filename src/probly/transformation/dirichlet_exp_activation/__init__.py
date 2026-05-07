"""Dirichlet exp-activation transformation."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE
from probly.transformation.dirichlet_exp_activation import _common
from probly.transformation.dirichlet_exp_activation._common import DirichletExpActivationPredictor

dirichlet_exp_activation = _common.dirichlet_exp_activation
register = _common.register


## Torch
@_common.dirichlet_exp_activation_appender.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@_common.dirichlet_exp_activation_appender.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = ["DirichletExpActivationPredictor", "dirichlet_exp_activation", "register"]
