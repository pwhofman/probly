"""JAX backend tests for the selective prediction metrics."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from probly.metrics.selective_prediction import (  # noqa: E402
    augrc,
    aurc,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)


def _parity_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return np.round(rng.random(100), 1).astype(np.float32), (rng.random(100) < 0.3).astype(np.float32)


def test_risk_coverage_curve_matches_numpy() -> None:
    criterion, losses = _parity_inputs()
    expected = risk_coverage_curve(criterion, losses)
    actual = risk_coverage_curve(jnp.asarray(criterion), jnp.asarray(losses))
    for e, a in zip(expected, actual, strict=True):
        assert isinstance(a, jax.Array)
        np.testing.assert_allclose(np.asarray(a), e, rtol=1e-6)


def test_metrics_match_numpy() -> None:
    criterion, losses = _parity_inputs()
    criterion_j, losses_j = jnp.asarray(criterion), jnp.asarray(losses)
    for function in (aurc, augrc):
        np.testing.assert_allclose(float(function(criterion_j, losses_j)), function(criterion, losses), rtol=1e-6)
    for coverage in (0.25, 0.5, 1.0):
        actual = risk_at_coverage(criterion_j, losses_j, coverage)
        np.testing.assert_allclose(float(actual), risk_at_coverage(criterion, losses, coverage), rtol=1e-6)
    for risk in (0.0, 0.2, 0.3):
        actual = coverage_at_risk(criterion_j, losses_j, risk)
        np.testing.assert_allclose(float(actual), coverage_at_risk(criterion, losses, risk), rtol=1e-6)


def test_risk_coverage_curve_and_aurc_under_jit() -> None:
    criterion, losses = _parity_inputs()
    criterion_j, losses_j = jnp.asarray(criterion), jnp.asarray(losses)
    coverage, risk, _ = jax.jit(risk_coverage_curve)(criterion_j, losses_j)
    expected_coverage, expected_risk, _ = risk_coverage_curve(criterion, losses)
    np.testing.assert_allclose(np.asarray(coverage), expected_coverage, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(risk), expected_risk, rtol=1e-6)
    np.testing.assert_allclose(float(jax.jit(aurc)(criterion_j, losses_j)), aurc(criterion, losses), rtol=1e-6)


def test_metrics_under_vmap() -> None:
    criterion, losses = _parity_inputs()
    batched = jax.vmap(augrc)(jnp.stack([criterion, criterion[::-1]]), jnp.stack([losses, losses[::-1]]))
    np.testing.assert_allclose(np.asarray(batched), [augrc(criterion, losses)] * 2, rtol=1e-6)


def test_risk_coverage_curve_accepts_numpy_losses() -> None:
    coverage, _, _ = risk_coverage_curve(jnp.array([0.5, 0.1]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(np.asarray(coverage), [0.0, 0.5, 1.0])


def test_nan_criterion_raises() -> None:
    with pytest.raises(ValueError, match="criterion must not contain NaN"):
        aurc(jnp.array([0.1, jnp.nan]), jnp.zeros(2))


def test_nan_losses_raise() -> None:
    with pytest.raises(ValueError, match="losses must not contain NaN"):
        aurc(jnp.array([0.1, 0.2]), jnp.array([0.0, jnp.nan]))
