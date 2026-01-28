"""Helpful mathematical functions for BBQ Calibration."""

from __future__ import annotations

import torch
from torch import Tensor


def betaln(a: Tensor, b: Tensor) -> Tensor:
    """Natural log of the Beta Function."""
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
