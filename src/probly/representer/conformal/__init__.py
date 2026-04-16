"""This module contains the conformal representers for regression and classification."""

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    ConformalRepresenter,
    ensure1d,
    ensure2d,
)


@ensure1d.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ensure2d.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "ConformalRepresenter",
]
