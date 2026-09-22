"""Torch implementations of point deciders."""

from __future__ import annotations

import torch

from ._common import point_from_mean


@point_from_mean.register(torch.Tensor)
def _(prediction: torch.Tensor) -> torch.Tensor:
    return prediction
