"""Torch quantification of DUQ representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.representation.duq.torch import TorchDUQRepresentation

from ._common import duq_uncertainty

if TYPE_CHECKING:
    import torch


@duq_uncertainty.register(TorchDUQRepresentation)
def torch_duq_uncertainty(representation: TorchDUQRepresentation) -> torch.Tensor:
    r"""Per-sample DUQ uncertainty :math:`1 - \max_c K_c(x)` for torch tensors."""
    return 1.0 - representation.kernel_values.max(dim=-1).values
