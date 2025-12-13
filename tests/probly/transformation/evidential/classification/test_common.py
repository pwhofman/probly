"""Tests for evidential classification registration and dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from probly.transformation.evidential.classification.common import (
    evidential_classification,
    register,
)


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
