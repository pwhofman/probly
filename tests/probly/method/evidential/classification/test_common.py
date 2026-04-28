"""Tests for evidential classification registration and dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from probly.method.evidential.classification import (
    evidential_classification,
    register,
)


def test_multiple_types_are_handled_independently() -> None:
    class EnhancedModel:
        def __init__(self, value: str) -> None:
            self.value = value

    class ModelA:
        def __init__(self, mid: str) -> None:
            self.id = mid

    class ModelB:
        def __init__(self, mid: str) -> None:
            self.id = mid

    def appender_a(base: ModelA) -> EnhancedModel:
        return EnhancedModel(f"A_Enhanced({base.id})")

    def appender_b(base: ModelB) -> EnhancedModel:
        return EnhancedModel(f"B_Enhanced({base.id})")

    register(ModelA, cast(Callable[..., object], appender_a))
    register(ModelB, cast(Callable[..., object], appender_b))

    a = ModelA("001")
    b = ModelB("002")

    result_a = evidential_classification(cast(Any, a))
    result_b = evidential_classification(cast(Any, b))

    assert cast(EnhancedModel, result_a).value == "A_Enhanced(001)"
    assert cast(EnhancedModel, result_b).value == "B_Enhanced(002)"
