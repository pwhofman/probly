"""NumPy backend tests for ``probly.evaluation`` conformal-set metrics."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.conformal_set.numpy import NumpyIntervalConformalSet, NumpyOneHotConformalSet

from ._metrics_suite import MetricsSuite


@pytest.fixture
def array_fn():
    return np.asarray


@pytest.fixture
def make_onehot_set():
    return lambda mask: NumpyOneHotConformalSet(array=np.asarray(mask))


@pytest.fixture
def make_interval_set():
    return lambda intervals: NumpyIntervalConformalSet(array=np.asarray(intervals))


class TestNumpy(MetricsSuite):
    """NumPy implementation of the shared evaluation suite."""
