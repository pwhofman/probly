"""Proper scoring rules for uncertainty quantification."""

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from . import array as array  # eager numpy registration
from ._common import (
    BrierLoss,
    LogLoss,
    ProperScoringRule,
    SphericalLoss,
    ZeroOneLoss,
    _brier_loss_vector,
    _log_loss_vector,
    _spherical_loss_vector,
    _zero_one_loss_vector,
)


@_log_loss_vector.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@_brier_loss_vector.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@_zero_one_loss_vector.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@_spherical_loss_vector.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "BrierLoss",
    "LogLoss",
    "ProperScoringRule",
    "SphericalLoss",
    "ZeroOneLoss",
]
