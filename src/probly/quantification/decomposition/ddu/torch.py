"""Torch DDU uncertainty decomposition helpers."""

from __future__ import annotations

import torch

from ._common import negative_log_density


@negative_log_density.register(torch.Tensor)
def torch_negative_log_density(densities: torch.Tensor) -> torch.Tensor:
    """Convert class-weighted log densities to negative GMM log density."""
    return -torch.logsumexp(densities, dim=-1)
