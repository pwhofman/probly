"""NumPy implementation of the selective prediction metrics."""

from __future__ import annotations

import numpy as np

from ._common import (
    augrc,
    aurc,
    check_coverage,
    check_inputs,
    check_no_nan,
    check_risk,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)


def _numpy_sorted_runs(
    criterion: np.ndarray, losses: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort by the criterion and find the runs of tied criterion values.

    Returns:
        The sorted criterion, the sorted losses (as floats), and for every position the first and the last
        position of its run.
    """
    losses = np.asarray(losses, dtype=float)
    n = check_inputs(criterion, losses)
    check_no_nan(np.isnan(criterion).any())
    order = np.argsort(criterion, kind="stable")
    criterion_sorted = criterion[order]
    losses_sorted = losses[order]

    # Ties are detected in the dtype of the criterion, since a cast to the dtype of the losses could merge
    # distinct values.
    is_new = criterion_sorted[1:] != criterion_sorted[:-1]
    is_run_start = np.concatenate([[True], is_new])
    is_run_end = np.concatenate([is_new, [True]])
    run_start = np.maximum.accumulate(np.where(is_run_start, np.arange(n), 0))
    run_end = np.flip(np.minimum.accumulate(np.flip(np.where(is_run_end, np.arange(n), n - 1))))
    return criterion_sorted, losses_sorted, run_start, run_end


@risk_coverage_curve.register(np.ndarray)
def numpy_risk_coverage_curve(criterion: np.ndarray, losses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the exact risk-coverage curve for NumPy arrays."""
    criterion_sorted, losses_sorted, _, run_end = _numpy_sorted_runs(criterion, losses)
    n = len(losses_sorted)

    # Tied instances are accepted together: every position in a run of tied criterion values points to the end
    # of that run, as in probly.metrics.numpy._binary_clf_curve.
    count = run_end + 1
    risk = np.cumsum(losses_sorted)[run_end] / count
    coverage = np.concatenate([[0.0], count / n])
    risk = np.concatenate([risk[:1], risk])
    thresholds = np.concatenate([[-np.inf], criterion_sorted.astype(float)])
    return coverage, risk, thresholds


@aurc.register(np.ndarray)
def numpy_aurc(criterion: np.ndarray, losses: np.ndarray) -> float:
    """Compute the area under the exact risk-coverage curve for NumPy arrays."""
    _, losses_sorted, run_start, run_end = _numpy_sorted_runs(criterion, losses)
    n = len(losses_sorted)

    # Accepting the k most confident instances, with a random part of a tied run, gives an expected cumulative
    # loss that is linear in k inside the run. The expected selective risk at every k is that loss over k.
    cumulative = np.cumsum(losses_sorted)
    before_run = np.where(run_start > 0, cumulative[run_start - 1], 0.0)
    count = np.arange(1, n + 1)
    expected_loss = before_run + (cumulative[run_end] - before_run) * (count - run_start) / (run_end - run_start + 1)
    risk = expected_loss / count
    return float(np.trapezoid(np.concatenate([risk[:1], risk]), np.concatenate([[0.0], count / n])))


@augrc.register(np.ndarray)
def numpy_augrc(criterion: np.ndarray, losses: np.ndarray) -> float:
    """Compute the area under the exact generalized risk-coverage curve for NumPy arrays."""
    coverage, risk, _ = numpy_risk_coverage_curve(criterion, losses)
    return float(np.trapezoid(risk * coverage, coverage))


@risk_at_coverage.register(np.ndarray)
def numpy_risk_at_coverage(criterion: np.ndarray, losses: np.ndarray, coverage: float) -> float:
    """Compute the selective risk at a target coverage for NumPy arrays."""
    check_coverage(coverage)
    curve_coverage, risk, _ = numpy_risk_coverage_curve(criterion, losses)
    # The coverage is non-decreasing, so the first point that reaches the target has the smallest coverage.
    return float(risk[np.argmax(curve_coverage >= coverage)])


@coverage_at_risk.register(np.ndarray)
def numpy_coverage_at_risk(criterion: np.ndarray, losses: np.ndarray, risk: float) -> float:
    """Compute the largest coverage with at most a target selective risk for NumPy arrays."""
    check_risk(risk)
    coverage, curve_risk, _ = numpy_risk_coverage_curve(criterion, losses)
    # The endpoint at coverage 0 is excluded, since it accepts no instance.
    return float(np.max(np.where(curve_risk[1:] <= risk, coverage[1:], 0.0)))
