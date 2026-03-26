"""Efficient credal prediction method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from . import common

efficient_credal_prediction = common.efficient_credal_prediction


## Torch
@common.efficient_credal_prediction_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415
