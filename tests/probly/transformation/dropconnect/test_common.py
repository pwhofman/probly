"""Simple test for common dropconnect functionality."""

from __future__ import annotations

from unittest.mock import Mock

from probly.transformation.dropconnect import dropconnect


def test_dropconnect_common() -> None:
    """Test that dropconnect function exists and works with mock models."""
    # Test that function exists and is callable
    assert callable(dropconnect)

    # Test with mock model
    mock_model = Mock()
    result = dropconnect(mock_model, p=0.5)

    # Should return something without crashing
    assert result is not None
