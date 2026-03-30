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
