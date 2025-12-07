"""Tests for the general dropconnect transformation API."""

from __future__ import annotations

from probly.transformation.dropconnect import dropconnect


def test_dropconnect_exists() -> None:
    """Test that the dropconnect function can be imported."""
    assert callable(dropconnect)
