"""Tests for stream builders."""

from __future__ import annotations

import pytest

from river_uq.streams import build_stream


@pytest.mark.parametrize("name", ["stagger_drift", "sea_drift", "agrawal_stationary"])
def test_build_stream_yields_pairs(name: str) -> None:
    it, _ = build_stream(name, seed=0, n=50)
    pairs = list(it)
    assert len(pairs) == 50
    for x, y in pairs:
        assert isinstance(x, dict)
        assert all(isinstance(v, (int, float)) for v in x.values())
        assert y in (0, 1, 2, 3)  # all our streams are 2-class or up to 4-class


def test_drift_streams_report_drift_t() -> None:
    _, t_stagger = build_stream("stagger_drift", seed=0, n=10)
    _, t_sea = build_stream("sea_drift", seed=0, n=10)
    _, t_agra = build_stream("agrawal_stationary", seed=0, n=10)
    assert t_stagger == 2000
    assert t_sea == 2000
    assert t_agra is None


def test_unknown_stream_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_stream("nope", seed=0)
