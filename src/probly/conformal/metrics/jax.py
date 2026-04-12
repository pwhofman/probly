"""JAX implementations of conformal prediction metrics."""

from __future__ import annotations

import jax.numpy as jnp

from ._common import (
    average_interval_size,
    average_set_size,
    empirical_coverage_classification,
    empirical_coverage_regression,
)


@empirical_coverage_classification.register(jnp.ndarray)
def _empirical_coverage_classification_jax(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> float:
    contained = y_pred[jnp.arange(len(y_true)), y_true.astype(int)]
    return contained.mean().item()


@empirical_coverage_regression.register(jnp.ndarray)
def _empirical_coverage_regression_jax(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> float:
    return ((y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 1])).mean().item()


@average_set_size.register(jnp.ndarray)
def _average_set_size_jax(y_pred: jnp.ndarray) -> float:
    return y_pred.sum(axis=1).mean().item()


@average_interval_size.register(jnp.ndarray)
def _average_interval_size_jax(y_pred: jnp.ndarray) -> float:
    return (y_pred[:, 1] - y_pred[:, 0]).mean().item()
