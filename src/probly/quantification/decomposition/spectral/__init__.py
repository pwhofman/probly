"""Spectral uncertainty decompositions."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE
from probly.quantification._quantification import decompose


@decompose.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


def __getattr__(name: str) -> object:
    if name in {"SpectralDecomposition", "spectral_decomposition"}:
        from . import torch as spectral_torch  # noqa: PLC0415

        return getattr(spectral_torch, name)
    raise AttributeError(name)


__all__ = ["SpectralDecomposition", "spectral_decomposition"]
