"""Tests for utils.sets functions."""

from __future__ import annotations

from probly.utils.sets import powerset


def test_powerset() -> None:
    assert powerset([]) == [()]
    assert powerset([1]) == [(), (1,)]
    assert powerset([1, 2]) == [(), (1,), (2,), (1, 2)]
