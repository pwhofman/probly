"""Backend-agnostic tests: nauc and loop."""

from __future__ import annotations

import dataclasses
import math
import subprocess
import sys

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


def test_nauc_with_nan_entries():
    """NaN entries should be excluded while preserving x-axis spacing."""
    result = compute_nauc([0.5, float("nan"), 0.7, 0.9])
    assert 0.0 <= result <= 1.0
    assert not math.isnan(result)


# ---------------------------------------------------------------------------
# ALState
# ---------------------------------------------------------------------------


def test_alstate_is_dataclass():
    assert dataclasses.is_dataclass(ALState)
    field_names = {f.name for f in dataclasses.fields(ALState)}
    assert field_names == {"iteration", "pool", "estimator"}


def test_numpy_metrics_do_not_import_torch() -> None:
    code = """
import sys
import numpy as np
from probly.evaluation.active_learning.metrics import compute_accuracy, compute_ece

assert 'torch' not in sys.modules
assert 'probly.evaluation.active_learning.torch_metrics' not in sys.modules
labels = np.array([0, 1])
assert compute_accuracy(labels, labels) == 1.0
assert compute_ece(np.eye(2), labels) == 0.0
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr
