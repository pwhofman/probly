"""Tests for evidential classification registration and dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from probly.transformation.evidential.classification.common import (
    evidential_classification,
    register,
)


# === Test 1: Kein Appender registriert -> Fehler ===
def test_evidential_classification_raises_not_implemented_error() -> None:
    class DummyPredictor:
        """Minimal predictor without appender."""

    dummy = DummyPredictor()

    with pytest.raises(NotImplementedError) as excinfo:
        # cast -> mypy fix: DummyPredictor not bound to T
        evidential_classification(cast(Any, dummy))

    assert "No evidential classification appender registered" in str(excinfo.value)


# === Test 2: Ein registrierter Appender wird korrekt aufgerufen ===
def test_registered_appender_is_called() -> None:
    class DummyPredictor:
        def __init__(self, name: str) -> None:
            self.name = name

    def dummy_appender(base: DummyPredictor) -> str:
        """Simulated evidential wrapper for DummyPredictor."""
        return f"Evidential({base.name})"

    register(DummyPredictor, cast(Callable[..., object], dummy_appender))

    dummy = DummyPredictor("ModelX")
    result = evidential_classification(cast(Any, dummy))

    assert cast(str, result) == "Evidential(ModelX)"


# === Test 3: Mehrere Typen können unabhängig registriert werden ===
def test_multiple_types_are_handled_independently() -> None:
    class ModelA:
        def __init__(self, mid: str) -> None:
            self.id = mid

    class ModelB:
        def __init__(self, mid: str) -> None:
            self.id = mid

    def appender_a(base: ModelA) -> str:
        return f"A_Enhanced({base.id})"

    def appender_b(base: ModelB) -> str:
        return f"B_Enhanced({base.id})"

    register(ModelA, cast(Callable[..., object], appender_a))
    register(ModelB, cast(Callable[..., object], appender_b))

    a = ModelA("001")
    b = ModelB("002")

    result_a = evidential_classification(cast(Any, a))
    result_b = evidential_classification(cast(Any, b))

    assert cast(str, result_a) == "A_Enhanced(001)"
    assert cast(str, result_b) == "B_Enhanced(002)"
