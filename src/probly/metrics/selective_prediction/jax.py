"""JAX implementation of the selective prediction metrics."""

from __future__ import annotations

import jax
from jax.core import Tracer
import jax.numpy as jnp

from ._common import (
    augrc,
    aurc,
    check_coverage,
    check_inputs,
    check_no_nan,
    check_no_negative_infinity,
    check_risk,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)


def _jax_sorted_runs(criterion: jax.Array, losses: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sort by the criterion and find the runs of tied criterion values.

    Returns:
        The sorted criterion, the sorted losses (as floats), and for every position the first and the last
        position of its run.
    """
    losses = jnp.asarray(losses, dtype=jnp.result_type(float))
    n = check_inputs(criterion, losses)
    # Inside a traced function the values are unknown, so the check can only run on concrete arrays.
    if not isinstance(criterion, Tracer):
        check_no_nan(jnp.isnan(criterion).any())
        check_no_negative_infinity(jnp.isneginf(criterion).any())
    if not isinstance(losses, Tracer):
        check_no_nan(jnp.isnan(losses).any(), "losses")
    order = jnp.argsort(criterion, stable=True)
    criterion_sorted = criterion[order]
    losses_sorted = losses[order]

    # Ties are detected in the dtype of the criterion, since a cast to the dtype of the losses could merge
    # distinct values.
    is_new = criterion_sorted[1:] != criterion_sorted[:-1]
    true = jnp.ones(1, dtype=bool)
    positions = jnp.arange(n)
    run_start = jax.lax.cummax(jnp.where(jnp.concatenate([true, is_new]), positions, 0), axis=0)
    run_end = jnp.where(jnp.concatenate([is_new, true]), positions, n - 1)
    run_end = jax.lax.cummin(run_end, axis=0, reverse=True)
    return criterion_sorted, losses_sorted, run_start, run_end


@risk_coverage_curve.register((jax.Array, Tracer))
def jax_risk_coverage_curve(criterion: jax.Array, losses: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the exact risk-coverage curve for JAX arrays."""
    criterion_sorted, losses_sorted, _, run_end = _jax_sorted_runs(criterion, losses)
    n = len(losses_sorted)

    # Tied instances are accepted together: every position in a run of tied criterion values points to the end
    # of that run, as in probly.metrics.jax._binary_clf_curve.
    count = run_end + 1
    risk = jnp.cumsum(losses_sorted)[run_end] / count
    coverage = jnp.concatenate([jnp.zeros(1, dtype=losses_sorted.dtype), count / n])
    risk = jnp.concatenate([risk[:1], risk])
    threshold_dtype = criterion.dtype if jnp.issubdtype(criterion.dtype, jnp.floating) else losses_sorted.dtype
    thresholds = jnp.concatenate(
        [jnp.full(1, -jnp.inf, dtype=threshold_dtype), criterion_sorted.astype(threshold_dtype)]
    )
    return coverage, risk, thresholds


@aurc.register((jax.Array, Tracer))
def jax_aurc(criterion: jax.Array, losses: jax.Array) -> jax.Array:
    """Compute the area under the exact risk-coverage curve for JAX arrays."""
    _, losses_sorted, run_start, run_end = _jax_sorted_runs(criterion, losses)
    n = len(losses_sorted)

    # Accepting the k most confident instances, with a random part of a tied run, gives an expected cumulative
    # loss that is linear in k inside the run. The expected selective risk at every k is that loss over k.
    cumulative = jnp.cumsum(losses_sorted)
    before_run = jnp.where(run_start > 0, cumulative[run_start - 1], 0.0)
    count = jnp.arange(1, n + 1)
    expected_loss = before_run + (cumulative[run_end] - before_run) * (count - run_start) / (run_end - run_start + 1)
    risk = expected_loss / count
    coverage = jnp.concatenate([jnp.zeros(1, dtype=risk.dtype), count / n])
    return jnp.trapezoid(jnp.concatenate([risk[:1], risk]), coverage)


@augrc.register((jax.Array, Tracer))
def jax_augrc(criterion: jax.Array, losses: jax.Array) -> jax.Array:
    """Compute the area under the exact generalized risk-coverage curve for JAX arrays."""
    coverage, risk, _ = jax_risk_coverage_curve(criterion, losses)
    return jnp.trapezoid(risk * coverage, coverage)


@risk_at_coverage.register((jax.Array, Tracer))
def jax_risk_at_coverage(criterion: jax.Array, losses: jax.Array, coverage: float) -> tuple[jax.Array, jax.Array]:
    """Compute the selective risk at a target coverage for JAX arrays."""
    check_coverage(coverage)
    curve_coverage, risk, _ = jax_risk_coverage_curve(criterion, losses)
    # The coverage is non-decreasing, so the first point that reaches the target has the smallest coverage.
    index = jnp.argmax(curve_coverage >= coverage)
    return risk[index], curve_coverage[index]


@coverage_at_risk.register((jax.Array, Tracer))
def jax_coverage_at_risk(criterion: jax.Array, losses: jax.Array, risk: float) -> tuple[jax.Array, jax.Array]:
    """Compute the largest coverage with at most a target selective risk for JAX arrays."""
    check_risk(risk)
    coverage, curve_risk, _ = jax_risk_coverage_curve(criterion, losses)
    # The endpoint at coverage 0 is excluded, since it accepts no instance.
    feasible_coverage = jnp.where(curve_risk[1:] <= risk, coverage[1:], 0.0)
    index = jnp.argmax(feasible_coverage)
    best = feasible_coverage[index]
    return best, jnp.where(best > 0, curve_risk[1:][index], jnp.nan)
