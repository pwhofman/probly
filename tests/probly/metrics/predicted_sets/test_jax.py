"""JAX backend tests for conformal-set metrics."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
from jax import numpy as jnp

from probly.representation.conformal_set.jax import JaxArrayIntervalConformalSet, JaxArrayOneHotConformalSet

from ._metrics_suite import MetricsSuite


@pytest.fixture
def array_fn():
    return jnp.asarray


@pytest.fixture
def make_onehot_set():
    return lambda mask: JaxArrayOneHotConformalSet(array=jnp.asarray(mask))


@pytest.fixture
def make_interval_set():
    return lambda intervals: JaxArrayIntervalConformalSet(array=jnp.asarray(intervals))


class TestJax(MetricsSuite):
    """JAX implementation of the shared conformal-set metric suite."""
