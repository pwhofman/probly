"""Torch representers for conformal sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    import torch

from ._common import ensure1d, ensure2d


@ensure2d.register(TorchSample)
def _(prediction: TorchSample) -> torch.Tensor:
    """Ensure that the prediction has a sample dimension and a data dimension."""
    data_t = prediction.tensor
    if data_t.ndim == 1:
        return data_t
    if data_t.ndim == 2:
        return data_t
    if data_t.ndim == 3:
        return data_t.mean(dim=0)
    msg = "Predictions with more than 3 dimensions are not supported for conformal prediction."
    raise ValueError(msg)


@ensure1d.register(TorchSample)
def _(prediction: TorchSample) -> torch.Tensor:
    """Ensure that the prediction has a single data dimension."""
    data_t = prediction.tensor
    if data_t.ndim == 1:
        return data_t
    if data_t.ndim == 2:
        return data_t.mean(dim=0)
    msg = "Predictions with more than 2 dimensions are not supported for conformal regression."
    raise ValueError(msg)
