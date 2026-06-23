"""Spectral uncertainty measures."""

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import spectral_entropy


@spectral_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "spectral_entropy",
]
