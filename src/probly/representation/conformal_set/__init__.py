"""Conformal set representations."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import ConformalSet, OneHotConformalSet, create_interval_conformal_set, create_onehot_conformal_set
from .array import ArrayIntervalConformalSet, ArrayOneHotConformalSet


@create_onehot_conformal_set.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@create_interval_conformal_set.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "ArrayIntervalConformalSet",
    "ArrayOneHotConformalSet",
    "ConformalSet",
    "OneHotConformalSet",
    "create_interval_conformal_set",
    "create_onehot_conformal_set",
]
