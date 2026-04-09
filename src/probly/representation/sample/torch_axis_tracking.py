"""Axis-tracking for PyTorch tensors."""

from __future__ import annotations

import torch

from probly.representation.sample.axis_tracking import ArrayIndex, convert_idx


@convert_idx.register(torch.Tensor)
def _convert_torch_tensor_idx(idx: torch.Tensor) -> ArrayIndex | bool | int:
    if idx.ndim == 0:
        if idx.dtype == torch.bool:
            return bool(idx)
        return 0
    return ArrayIndex(ndim=idx.ndim, is_boolean=idx.dtype == torch.bool)
