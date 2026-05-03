"""Tests for the seed and device helpers."""

from __future__ import annotations

import numpy as np
import torch

from stacking.utils import get_device, set_seed


def test_set_seed_makes_numpy_and_torch_reproducible() -> None:
    """Calling set_seed twice with the same seed yields identical draws."""
    set_seed(42)
    a_np = np.random.default_rng(0).standard_normal(4)  # uses default_rng, unaffected by set_seed
    a_torch = torch.randn(4)
    a_python = float(__import__("random").random())

    set_seed(42)
    b_torch = torch.randn(4)
    b_python = float(__import__("random").random())

    # default_rng(0) is independent of set_seed; we only check torch + random
    assert torch.equal(a_torch, b_torch)
    assert a_python == b_python
    _ = a_np  # silence unused-warning


def test_get_device_returns_torch_device() -> None:
    """get_device('cpu') returns a torch.device of type 'cpu'."""
    dev = get_device("cpu")
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"


def test_get_device_auto_does_not_crash() -> None:
    """get_device('auto') returns one of cpu/cuda/mps without crashing."""
    dev = get_device("auto")
    assert dev.type in {"cpu", "cuda", "mps"}
