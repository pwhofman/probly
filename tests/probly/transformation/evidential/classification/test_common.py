# ruff: noqa: D107

"""Tests for shared evidential classification dispatcher (common.py)."""

from __future__ import annotations

import pytest

from probly.transformation.evidential.classification.common import (
    evidential_classification,
    evidential_classification_appender,
    register,
)


class DummyPredictor:  # base model
    def __call__(self) -> None: ...


class WrappedPredictor:
    """Simple wrapper to verify the appender was applied."""

    def __init__(self, base: DummyPredictor) -> None:
        self.base = base


def dummy_appender(base: DummyPredictor) -> WrappedPredictor:
    """Appender used in tests; wraps the base predictor."""
    return WrappedPredictor(base)


def test_unregistered_type_raises_not_implemented() -> None:
    """Calling evidential_classification on an unregistered type must raise."""
    base = DummyPredictor()
    with pytest.raises(NotImplementedError) as exc:
        _ = evidential_classification(base)
    assert type(base).__name__ in str(exc.value)


def test_register_and_dispatch_wraps_base() -> None:
    """After registering, dispatch should call the appender and return its result."""
    register(DummyPredictor, dummy_appender)  # type:ignore[arg-type]

    base = DummyPredictor()
    out = evidential_classification(base)

    assert isinstance(out, WrappedPredictor)
    assert out.base is base


def test_registration_on_base_type_works_for_subclasses() -> None:
    """Registering for a base class must also handle subclass instances."""

    class ChildPredictor(DummyPredictor):
        pass

    child = ChildPredictor()
    out = evidential_classification(child)

    assert isinstance(out, WrappedPredictor)
    assert out.base is child


def test_register_returns_none_and_does_not_raise() -> None:
    """register() itself should be side-effect-only and return None."""

    def another_appender(base: DummyPredictor) -> WrappedPredictor:
        return WrappedPredictor(base)

    register(DummyPredictor, another_appender)


def test_direct_appender_call_matches_dispatch() -> None:
    """Calling the dispatcher directly equals evidential_classification for registered types."""
    base = DummyPredictor()
    register(DummyPredictor, dummy_appender)

    via_api = evidential_classification(base)
    via_dispatch: WrappedPredictor = evidential_classification_appender(base)

    assert isinstance(via_api, WrappedPredictor)
    assert isinstance(via_dispatch, WrappedPredictor)
    assert via_api.base is base
    assert via_dispatch.base is base
