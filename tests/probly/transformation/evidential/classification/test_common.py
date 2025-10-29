"""Tests for shared evidential classification dispatcher (common.py)."""

from __future__ import annotations

import pytest

from probly.transformation.evidential.classification.common import (
    evidential_classification,
    evidential_classification_appender,
    register,
)


class _DummyPredictor:
    """Minimal stand-in for a Predictor at runtime (no behavior needed)."""


class _WrappedPredictor:
    """Simple wrapper to verify the appender was applied."""

    def __init__(self, base: _DummyPredictor) -> None:
        self.base = base


def _dummy_appender(base: _DummyPredictor) -> _WrappedPredictor:
    """Appender used in tests; wraps the base predictor."""
    return _WrappedPredictor(base)


def test_unregistered_type_raises_not_implemented() -> None:
    """Calling evidential_classification on an unregistered type must raise."""
    base = _DummyPredictor()
    with pytest.raises(NotImplementedError) as exc:
        _ = evidential_classification(base)
    assert type(base).__name__ in str(exc.value)


def test_register_and_dispatch_wraps_base() -> None:
    """After registering, dispatch should call the appender and return its result."""
    register(_DummyPredictor, _dummy_appender)

    base = _DummyPredictor()
    out = evidential_classification(base)

    assert isinstance(out, _WrappedPredictor)
    assert out.base is base


def test_registration_on_base_type_works_for_subclasses() -> None:
    """Registering for a base class must also handle subclass instances."""

    class _ChildPredictor(_DummyPredictor):
        pass

    child = _ChildPredictor()
    out = evidential_classification(child)

    assert isinstance(out, _WrappedPredictor)
    assert out.base is child


def test_register_returns_none_and_does_not_raise() -> None:
    """register() itself should be side-effect-only and return None."""

    def _another_appender(base: _DummyPredictor) -> _WrappedPredictor:
        return _WrappedPredictor(base)

    result = register(_DummyPredictor, _another_appender)
    assert result is None


def test_direct_appender_call_matches_dispatch() -> None:
    """Calling the dispatcher directly equals evidential_classification for registered types."""
    base = _DummyPredictor()
    register(_DummyPredictor, _dummy_appender)

    via_api = evidential_classification(base)
    via_dispatch = evidential_classification_appender(base)

    assert isinstance(via_api, _WrappedPredictor)
    assert isinstance(via_dispatch, _WrappedPredictor)
    assert via_api.base is base
    assert via_dispatch.base is base
