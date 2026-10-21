"""Tests for probly utility dispatch fallbacks."""

from __future__ import annotations

import pytest

from probly.utils import entropy, intersection_probability


def test_utils_dispatchers_reject_unsupported_types() -> None:
    with pytest.raises(NotImplementedError, match="No entropy implementation"):
        entropy([0.5, 0.5])
    with pytest.raises(NotImplementedError, match="No intersection probability implementation"):
        intersection_probability([0.1], [0.9])
