"""PyTorch backend tests for the selective prediction metrics."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from probly.metrics.selective_prediction import (  # noqa: E402
    augrc,
    aurc,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)


def _parity_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return np.round(rng.random(100), 1), (rng.random(100) < 0.3).astype(float)


def test_risk_coverage_curve_matches_numpy() -> None:
    criterion, losses = _parity_inputs()
    expected = risk_coverage_curve(criterion, losses)
    actual = risk_coverage_curve(torch.from_numpy(criterion), torch.from_numpy(losses))
    for e, a in zip(expected, actual, strict=True):
        assert isinstance(a, torch.Tensor)
        np.testing.assert_allclose(a.numpy(), e)


def test_metrics_match_numpy() -> None:
    criterion, losses = _parity_inputs()
    criterion_t, losses_t = torch.from_numpy(criterion), torch.from_numpy(losses)
    for function in (aurc, augrc):
        np.testing.assert_allclose(float(function(criterion_t, losses_t)), function(criterion, losses))
    for coverage in (0.25, 0.5, 1.0):
        actual = risk_at_coverage(criterion_t, losses_t, coverage)
        np.testing.assert_allclose(float(actual), risk_at_coverage(criterion, losses, coverage))
    for risk in (0.0, 0.2, 0.3):
        actual = coverage_at_risk(criterion_t, losses_t, risk)
        np.testing.assert_allclose(float(actual), coverage_at_risk(criterion, losses, risk))


def test_risk_coverage_curve_casts_integer_losses() -> None:
    coverage, risk, thresholds = risk_coverage_curve(torch.tensor([0.5, 0.1]), torch.tensor([1, 0]))
    assert risk.is_floating_point()
    torch.testing.assert_close(risk, torch.tensor([0.0, 0.0, 0.5]))
    torch.testing.assert_close(coverage, torch.tensor([0.0, 0.5, 1.0]))
    assert thresholds[0] == -torch.inf


def test_risk_coverage_curve_finds_ties_before_casting() -> None:
    # The two smaller criterion values differ in float64 but are equal in float32, the dtype to which the
    # integer losses are cast.
    criterion = np.array([0.1, 0.1 + 1e-12, 0.9])
    losses = np.array([1, 0, 0])
    coverage, _, thresholds = risk_coverage_curve(torch.from_numpy(criterion), torch.from_numpy(losses))
    torch.testing.assert_close(coverage, torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]))
    assert thresholds.dtype == torch.float64
    np.testing.assert_allclose(
        float(aurc(torch.from_numpy(criterion), torch.from_numpy(losses))), aurc(criterion, losses)
    )


def test_risk_coverage_curve_half_precision_counts_exactly() -> None:
    # The count of 70000 exceeds the largest float16 value (65504), so the curve must be computed in float32.
    rng = np.random.default_rng(0)
    criterion = rng.random(70_000).astype(np.float16)
    losses = (rng.random(70_000) < 0.3).astype(np.float16)
    coverage, risk, _ = risk_coverage_curve(torch.from_numpy(criterion), torch.from_numpy(losses))
    assert coverage.dtype == risk.dtype == torch.float32
    assert coverage[-1] == 1.0
    np.testing.assert_allclose(
        float(aurc(torch.from_numpy(criterion), torch.from_numpy(losses))), aurc(criterion, losses), rtol=1e-4
    )


def test_risk_coverage_curve_accepts_numpy_losses() -> None:
    coverage, _, _ = risk_coverage_curve(torch.tensor([0.5, 0.1]), np.array([1.0, 0.0]))
    torch.testing.assert_close(coverage, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))


def test_nan_criterion_raises() -> None:
    with pytest.raises(ValueError, match="criterion must not contain NaN"):
        aurc(torch.tensor([0.1, torch.nan]), torch.zeros(2))


def test_nan_losses_raise() -> None:
    with pytest.raises(ValueError, match="losses must not contain NaN"):
        aurc(torch.tensor([0.1, 0.2]), torch.tensor([0.0, torch.nan]))


def test_negative_infinite_criterion_raises() -> None:
    with pytest.raises(ValueError, match="criterion must not contain -inf"):
        aurc(torch.tensor([-torch.inf, 0.2]), torch.zeros(2))
