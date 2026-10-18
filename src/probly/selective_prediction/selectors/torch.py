"""PyTorch support for selective prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._common import _to_float64_numpy

if TYPE_CHECKING:
    import numpy as np


@_to_float64_numpy.register(torch.Tensor)
def _torch_to_float64_numpy(uncertainty: torch.Tensor) -> np.ndarray:
    return uncertainty.detach().to("cpu", torch.float64).numpy()
