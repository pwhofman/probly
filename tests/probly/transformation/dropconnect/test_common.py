"""Simple test for common dropconnect functionality."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from probly.transformation.dropconnect import dropconnect


@pytest.mark.parametrize("p_value", [0.0, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0])
def test_dropconnect_different_probabilities(p_value: float) -> None:
    """Test dropconnect with various probability values."""
    # Test that function exists and is callable
    assert callable(dropconnect)

    # Test with mock model
    mock_model = Mock()
    result = dropconnect(mock_model, p=p_value)

    # Should return something without crashing
    assert result is not None
