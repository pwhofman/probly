"""Torch representers for conformal sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    import torch

from ._common import flatten_ensemble_quantile_sample, flatten_sample


@flatten_sample.register(TorchSample)
def _(sample: TorchSample) -> torch.Tensor:
    raw_tensor = sample.tensor
    if raw_tensor.ndim == 1:
        return raw_tensor
    # Check if the tensor is just badly shaped and flatten it if so, otherwise take the mean as normal.
    if raw_tensor.ndim == 2 and (raw_tensor.shape[1] == 1 or raw_tensor.shape[0] == 1):
        return raw_tensor.flatten()
    return sample.sample_mean()


@flatten_ensemble_quantile_sample.register(TorchSample)
def _(sample: TorchSample) -> torch.Tensor:
    raw_tensor = sample.tensor
    if raw_tensor.ndim == 3:
        return sample.sample_mean()
    return raw_tensor
