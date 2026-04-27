"""Backend-agnostic tests: nauc and loop."""

from __future__ import annotations

import dataclasses
import math

import pytest

from probly.evaluation.active_learning.loop import ALState
from probly.evaluation.active_learning.metrics import compute_nauc

# ---------------------------------------------------------------------------
# compute_nauc
# ---------------------------------------------------------------------------


def test_nauc_constant_one():
    assert compute_nauc([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_nauc_constant_below_one():
    assert compute_nauc([0.8, 0.8, 0.8, 0.8]) == pytest.approx(0.8)


def test_nauc_fast_improver_beats_slow():
    slow = compute_nauc([0.5, 0.5, 0.5, 0.8])
    fast = compute_nauc([0.5, 0.6, 0.7, 0.8])
    assert slow < fast


def test_nauc_result_in_unit_interval():
    nauc = compute_nauc([0.3, 0.5, 0.7, 0.9])
    assert 0.0 <= nauc <= 1.0


def test_nauc_single_value_is_nan():
    result = compute_nauc([0.7])
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# ALState
# ---------------------------------------------------------------------------


def test_alstate_is_dataclass():
    assert dataclasses.is_dataclass(ALState)
    field_names = {f.name for f in dataclasses.fields(ALState)}
    assert field_names == {"iteration", "pool", "estimator"}
