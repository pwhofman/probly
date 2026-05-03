"""Tests for the embedding cache I/O and registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from stacking.embed import ENCODERS, cache_path


def test_cache_path_format(tmp_path: Path) -> None:
    """cache_path joins root + '<dataset>_<encoder>_<split>.npz'."""
    p = cache_path(encoder="siglip2", dataset="cifar10h", split="test", root=tmp_path)
    assert p == tmp_path / "cifar10h_siglip2_test.npz"


def test_encoders_registry_lists_known_names() -> None:
    """ENCODERS exposes the two encoder names as registry keys."""
    assert set(ENCODERS) == {"siglip2", "dinov2_with_registers"}


def test_unknown_encoder_raises_keyerror() -> None:
    """Looking up an unknown encoder raises KeyError."""
    with pytest.raises(KeyError):
        _ = ENCODERS["bogus"]
