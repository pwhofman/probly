"""PyTorch backend tests for ``probly.evaluation`` conformal-set metrics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.representation.conformal_set.torch import (  # noqa: E402
    TorchIntervalConformalSet,
    TorchOneHotConformalSet,
)

from ._metrics_suite import MetricsSuite  # noqa: E402


def _to_tensor(values, *, dtype=None):
    if dtype is float:
        return torch.as_tensor(values, dtype=torch.float32)
    return torch.as_tensor(values)


@pytest.fixture
def array_fn():
    return _to_tensor


@pytest.fixture
def make_onehot_set():
    return lambda mask: TorchOneHotConformalSet(tensor=torch.as_tensor(mask))


@pytest.fixture
def make_interval_set():
    return lambda intervals: TorchIntervalConformalSet(tensor=torch.as_tensor(intervals))


class TestTorch(MetricsSuite):
    """PyTorch implementation of the shared evaluation suite."""
