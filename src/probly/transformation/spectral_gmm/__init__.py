"""Spectral-normalized Gaussian-mixture transformation for classification models."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import SpectralGMMPredictor, spectral_gmm, spectral_gmm_generator


## Torch
@spectral_gmm_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "SpectralGMMPredictor",
    "spectral_gmm",
    "spectral_gmm_generator",
]
