"""NumPy backend tests for ``probly.evaluation`` conformal-set metrics."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.conformal_set.array import ArrayIntervalConformalSet, ArrayOneHotConformalSet

from ._metrics_suite import MetricsSuite


@pytest.fixture
def array_fn():
    return np.asarray


@pytest.fixture
def make_onehot_set():
    return lambda mask: ArrayOneHotConformalSet(array=np.asarray(mask))


@pytest.fixture
def make_interval_set():
    return lambda intervals: ArrayIntervalConformalSet(array=np.asarray(intervals))


class TestNumpy(MetricsSuite):
    """NumPy implementation of the shared evaluation suite."""
