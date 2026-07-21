"""Seed and device helpers for the stacking playground."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy global RNG, and PyTorch.

    Note that NumPy code that uses ``np.random.default_rng(...)`` with an
    explicit seed is unaffected; this helper only seeds the global state.

    Args:
        seed: Integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    """Resolve a device name to a ``torch.device``.

    Args:
        name: One of ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``. The
            ``"auto"`` choice prefers CUDA, then MPS, then CPU.

    Returns:
        A ``torch.device``.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
