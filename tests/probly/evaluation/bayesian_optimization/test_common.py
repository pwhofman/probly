"""Backend-agnostic BO tests: metrics and the BOState dataclass."""

from __future__ import annotations

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from probly.evaluation.bayesian_optimization import (  # noqa: E402
    BOState,
    best_so_far,
    regret_curve,
    regret_nauc,
    simple_regret,
)

# ---------------------------------------------------------------------------
# best_so_far / simple_regret
# ---------------------------------------------------------------------------


def test_best_so_far_is_monotone_min():
    y = torch.tensor([3.0, 1.0, 5.0, 0.5, 4.0])
    assert torch.equal(best_so_far(y), torch.tensor([3.0, 1.0, 1.0, 0.5, 0.5]))


def test_simple_regret_nonnegative_and_zero_at_optimum():
    assert simple_regret(2.0, 0.5) == pytest.approx(1.5)
    assert simple_regret(-3.32, -3.32) == 0.0


# ---------------------------------------------------------------------------
# regret_curve / regret_nauc
# ---------------------------------------------------------------------------


def test_regret_curve_uses_running_minimum():
    y = torch.tensor([3.0, 1.0, 5.0, 0.5])
    curve = regret_curve(y, optimal_value=0.0)
    assert torch.equal(curve, torch.tensor([3.0, 1.0, 1.0, 0.5]))


def test_regret_nauc_in_unit_interval_for_decreasing_curve():
    y = torch.tensor([4.0, 3.0, 2.0, 1.0])
    nauc = regret_nauc(y, optimal_value=0.0)
    assert 0.0 < nauc < 1.0


def test_regret_nauc_zero_when_immediately_optimal():
    y = torch.zeros(5)
    nauc = regret_nauc(y, optimal_value=0.0)
    assert nauc == 0.0


def test_regret_nauc_nan_for_single_evaluation():
    nauc = regret_nauc(torch.tensor([5.0]), optimal_value=0.0)
    assert math.isnan(nauc)


# ---------------------------------------------------------------------------
# BOState
# ---------------------------------------------------------------------------


def test_bostate_is_dataclass_with_expected_fields():
    assert dataclasses.is_dataclass(BOState)
    field_names = {f.name for f in dataclasses.fields(BOState)}
    assert field_names == {"iteration", "x", "y", "best_y", "surrogate"}
