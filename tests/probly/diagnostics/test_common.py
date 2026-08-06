"""Backend-agnostic tests for individual diagnostic checks."""

from __future__ import annotations

import numpy as np

from probly.diagnostics import Verdict
from probly.diagnostics._diagnose import _check_uncertainty_variation


def test_constant_uncertainty_fails() -> None:
    assert _check_uncertainty_variation(np.full(100, 0.7)).verdict is Verdict.FAIL
    assert _check_uncertainty_variation(np.zeros(100)).verdict is Verdict.FAIL


def test_varying_uncertainty_passes() -> None:
    assert _check_uncertainty_variation(np.linspace(0.0, 1.0, 100)).verdict is Verdict.PASS
