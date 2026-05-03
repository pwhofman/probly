"""Bare-bones smoke test: stack_dare_temp_conformal.main() runs on each dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.stack_dare_temp_conformal as stack_script


def _write_synthetic_caches(root: Path, encoder: str, n_train: int, n_test: int, dim: int) -> None:
    """Mirror of helper from test_cifar10h_embeddings: tiny fake caches."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / f"cifar10_{encoder}_train.npz",
        X=rng.standard_normal((n_train, dim)).astype(np.float32),
        y_hard=rng.integers(0, 10, size=n_train).astype(np.int64),
    )
    counts = rng.integers(1, 51, size=(n_test, 10)).astype(np.float32)
    y_soft = counts / counts.sum(axis=1, keepdims=True)
    y_hard = y_soft.argmax(axis=1).astype(np.int64)
    np.savez(
        root / f"cifar10h_{encoder}_test.npz",
        X=rng.standard_normal((n_test, dim)).astype(np.float32),
        y_hard=y_hard,
        y_soft=y_soft,
        counts=counts,
    )


@pytest.mark.parametrize("dataset", ["two_moons", "cifar10h"])
def test_stack_dare_temp_conformal_smoke(dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The headline composition script runs end-to-end on tiny inputs."""
    if dataset == "cifar10h":
        encoder = "siglip2"
        _write_synthetic_caches(tmp_path, encoder=encoder, n_train=128, n_test=64, dim=8)
        from stacking.datasets import cifar10h_embeddings

        monkeypatch.setattr(cifar10h_embeddings, "_default_cache_root", lambda: tmp_path)

    argv = [
        "stack_dare_temp_conformal.py",
        "--dataset", dataset,
        "--epochs", "2",
        "--num-members", "2",
        "--seed", "0",
        "--device", "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    stack_script.main()
