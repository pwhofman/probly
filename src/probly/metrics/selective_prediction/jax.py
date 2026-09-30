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
    check_risk,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)


@risk_coverage_curve.register((jax.Array, Tracer))
def jax_risk_coverage_curve(criterion: jax.Array, losses: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the exact risk-coverage curve for JAX arrays."""
    losses = jnp.asarray(losses, dtype=jnp.result_type(float))
    n = check_inputs(criterion, losses)
    # Inside a traced function the values are unknown, so the check can only run on concrete arrays.
    if not isinstance(criterion, Tracer):
        check_no_nan(jnp.isnan(criterion).any())
    order = jnp.argsort(criterion, stable=True)
    criterion_sorted = criterion[order]
    losses_sorted = losses[order]

    # Tied instances are accepted together: every position in a run of tied criterion values points to the end
    # of that run, as in probly.metrics.jax._binary_clf_curve. Ties are detected in the dtype of the
    # criterion, since a cast to the dtype of the losses could merge distinct values.
    is_run_end = jnp.concatenate([criterion_sorted[1:] != criterion_sorted[:-1], jnp.ones(1, dtype=bool)])
    run_end = jnp.where(is_run_end, jnp.arange(n), n - 1)
    run_end = jax.lax.cummin(run_end, axis=0, reverse=True)

    count = run_end + 1
    risk = jnp.cumsum(losses_sorted)[run_end] / count
    coverage = jnp.concatenate([jnp.zeros(1, dtype=losses.dtype), count / n])
    risk = jnp.concatenate([risk[:1], risk])
    threshold_dtype = criterion.dtype if jnp.issubdtype(criterion.dtype, jnp.floating) else losses.dtype
    thresholds = jnp.concatenate(
        [jnp.full(1, -jnp.inf, dtype=threshold_dtype), criterion_sorted.astype(threshold_dtype)]
    )
    return coverage, risk, thresholds


@aurc.register((jax.Array, Tracer))
def jax_aurc(criterion: jax.Array, losses: jax.Array) -> jax.Array:
    """Compute the area under the exact risk-coverage curve for JAX arrays."""
    coverage, risk, _ = jax_risk_coverage_curve(criterion, losses)
    return jnp.trapezoid(risk, coverage)


@augrc.register((jax.Array, Tracer))
def jax_augrc(criterion: jax.Array, losses: jax.Array) -> jax.Array:
    """Compute the area under the exact generalized risk-coverage curve for JAX arrays."""
    coverage, risk, _ = jax_risk_coverage_curve(criterion, losses)
    return jnp.trapezoid(risk * coverage, coverage)


@risk_at_coverage.register((jax.Array, Tracer))
def jax_risk_at_coverage(criterion: jax.Array, losses: jax.Array, coverage: float) -> jax.Array:
    """Compute the selective risk at a target coverage for JAX arrays."""
    check_coverage(coverage)
    curve_coverage, risk, _ = jax_risk_coverage_curve(criterion, losses)
    # The coverage is non-decreasing, so the first point that reaches the target has the smallest coverage.
    return risk[jnp.argmax(curve_coverage >= coverage)]


@coverage_at_risk.register((jax.Array, Tracer))
def jax_coverage_at_risk(criterion: jax.Array, losses: jax.Array, risk: float) -> jax.Array:
    """Compute the largest coverage with at most a target selective risk for JAX arrays."""
    check_risk(risk)
    coverage, curve_risk, _ = jax_risk_coverage_curve(criterion, losses)
    # The endpoint at coverage 0 is excluded, since it accepts no instance.
    return jnp.max(jnp.where(curve_risk[1:] <= risk, coverage[1:], 0.0))
