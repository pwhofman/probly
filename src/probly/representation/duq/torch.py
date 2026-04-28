"""Torch DUQ representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected

from ._common import DUQRepresentation, create_duq_representation


@create_duq_representation.register(torch.Tensor)
@dataclass(frozen=True, slots=True)
class TorchDUQRepresentation(DUQRepresentation, TorchAxisProtected):
    """DUQ representation backed by a torch tensor.

    Args:
        kernel_values: Per-class RBF kernel values, shape ``(..., num_classes)``.
    """

    kernel_values: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"kernel_values": 1}
