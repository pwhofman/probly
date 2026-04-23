"""Uncertainty measures for credal sets."""

from probly.lazy_types import TORCH_TENSOR_LIKE

from ._common import (
    generalized_hartley,
    lower_entropy,
    upper_entropy,
)


@upper_entropy.delayed_register((TORCH_TENSOR_LIKE,))
@lower_entropy.delayed_register((TORCH_TENSOR_LIKE,))
@generalized_hartley.delayed_register((TORCH_TENSOR_LIKE,))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "generalized_hartley",
    "lower_entropy",
    "upper_entropy",
]
