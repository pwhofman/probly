"""Torch representers for conformal sets."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchTensorSample

from ._common import flatten_sample, flatten_ensemble_quantile_sample


@flatten_sample.register(TorchTensorSample)
def _(sample: TorchTensorSample) -> torch.Tensor:
    raw_tensor = sample.tensor
    if raw_tensor.ndim == 1:
        return raw_tensor
    # Check if the tensor is just badly shaped and flatten it if so, otherwise take the mean as normal.
    if raw_tensor.ndim == 2 and (raw_tensor.shape[1] == 1 or raw_tensor.shape[0] == 1):
        return raw_tensor.flatten()
    return sample.sample_mean()


@flatten_ensemble_quantile_sample.register(TorchTensorSample)
def _(sample: TorchTensorSample) -> torch.Tensor:
    raw_tensor = sample.tensor
    if raw_tensor.ndim == 3:
        return sample.sample_mean()
    return raw_tensor
