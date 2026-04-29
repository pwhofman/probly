"""Torch quantification of DUQ representations."""

from __future__ import annotations

import torch

from ._common import duq_uncertainty


@duq_uncertainty.register(torch.Tensor)
def torch_duq_uncertainty(kernel_values: torch.Tensor) -> torch.Tensor:
    r"""Per-sample DUQ uncertainty :math:`1 - \max_c K_c(x)` for torch tensors."""
    return 1.0 - kernel_values.max(dim=-1).values
