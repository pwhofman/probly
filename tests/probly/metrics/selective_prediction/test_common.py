"""Backend-agnostic tests for the selective prediction metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from probly.metrics.selective_prediction import (
    augrc,
    aurc,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("function", [risk_coverage_curve, aurc, augrc])
def test_unregistered_type_raises(function: Callable[..., object]) -> None:
    with pytest.raises(NotImplementedError, match="implementation registered"):
        function(object(), object())


@pytest.mark.parametrize("function", [risk_at_coverage, coverage_at_risk])
def test_working_point_unregistered_type_raises(function: Callable[..., object]) -> None:
    with pytest.raises(NotImplementedError, match="implementation registered"):
        function(object(), object(), 0.5)
