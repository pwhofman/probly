"""Collection of helper functions."""

from probly.lazy_types import TORCH_TENSOR_LIKE

from ._common import compute_mean_probs


## Torch
@compute_mean_probs.delayed_register(TORCH_TENSOR_LIKE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["compute_mean_probs"]
