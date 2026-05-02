"""Deciders for reducing representations to categorical distributions."""

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from . import array as array
from ._common import categorical_from_mean, mean_field_categorical


@categorical_from_mean.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["categorical_from_mean", "mean_field_categorical"]
