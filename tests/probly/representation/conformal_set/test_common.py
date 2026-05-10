"""Tests for the conformal-set common factory fallbacks."""

from __future__ import annotations

import pytest


class TestConformalSetCommonFallbacks:
    """The conformal-set factory functions raise for unregistered input types."""

    def test_create_onehot_raises(self) -> None:
        from probly.representation.conformal_set._common import create_onehot_conformal_set  # noqa: PLC0415

        with pytest.raises(NotImplementedError):
            create_onehot_conformal_set(object())

    def test_create_interval_raises(self) -> None:
        from probly.representation.conformal_set._common import create_interval_conformal_set  # noqa: PLC0415

        with pytest.raises(NotImplementedError):
            create_interval_conformal_set(object(), object())
