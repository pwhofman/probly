"""Spectral uncertainty measures."""

from probly.lazy_types import TORCH_VALUE

from ._common import conditional_spectral_entropy, spectral_entropy


@spectral_entropy.delayed_register(TORCH_VALUE)
@conditional_spectral_entropy.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "conditional_spectral_entropy",
    "spectral_entropy",
]
