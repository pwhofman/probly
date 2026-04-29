"""Representer for the efficient credal prediction method based on :cite:`hofmanefficient`."""

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import EfficientCredalRepresenter, compute_efficient_credal_bounds


## Torch
@compute_efficient_credal_bounds.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "EfficientCredalRepresenter",
    "compute_efficient_credal_bounds",
]
