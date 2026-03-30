"""Utils for benchmarking."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_id: int | None = None) -> torch.device:
    """Return the best available device, or a specific CUDA device if requested.

    Args:
        device_id: Optional CUDA device ID to use. If None, automatically selects the least utilized CUDA device.
            Ignored if CUDA is not available.

    Returns:
        The selected torch.device.

    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        try:
            us = [torch.cuda.utilization(i) for i in range(torch.cuda.device_count())]
            return torch.device(f"cuda:{us.index(min(us))}")
        except ModuleNotFoundError:
            return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
