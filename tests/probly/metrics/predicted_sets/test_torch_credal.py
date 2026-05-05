"""PyTorch credal-set tests for ``coverage`` and ``efficiency``.

Runs the shared :class:`CredalSuite` against the torch credal-set wrappers
and asserts numerical parity with the numpy backend on identical inputs.
Only ``TorchConvexCredalSet`` and ``TorchProbabilityIntervalsCredalSet``
are registered; ``TorchDistanceBasedCredalSet`` and
``TorchDirichletLevelSetCredalSet`` are intentionally not.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from probly.metrics import coverage, efficiency  # noqa: E402
from probly.representation.credal_set.array import (  # noqa: E402
    ArrayConvexCredalSet,
    ArrayProbabilityIntervalsCredalSet,
)
from probly.representation.credal_set.torch import (  # noqa: E402
    TorchConvexCredalSet,
    TorchDirichletLevelSetCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.array_categorical import ArrayProbabilityCategoricalDistribution  # noqa: E402
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution  # noqa: E402

from ._credal_suite import CredalSuite  # noqa: E402


@pytest.fixture
def make_convex():
    return lambda probs: TorchConvexCredalSet(
        tensor=TorchProbabilityCategoricalDistribution(torch.as_tensor(probs, dtype=torch.float64)),
    )


@pytest.fixture
def make_intervals():
    return lambda lower, upper: TorchProbabilityIntervalsCredalSet(
        lower_bounds=torch.as_tensor(lower, dtype=torch.float64),
        upper_bounds=torch.as_tensor(upper, dtype=torch.float64),
    )


@pytest.fixture
def make_distribution():
    return lambda probs: TorchProbabilityCategoricalDistribution(torch.as_tensor(probs, dtype=torch.float64))


class TestTorch(CredalSuite):
    """PyTorch implementation of the shared credal suite."""


def test_convex_numpy_torch_parity() -> None:
    probs = np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]])
    target_probs = np.array([[0.5, 0.4, 0.1]])
    np_cs = ArrayConvexCredalSet(array=ArrayProbabilityCategoricalDistribution(probs))
    tc_cs = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(torch.as_tensor(probs)))
    np_target = ArrayProbabilityCategoricalDistribution(target_probs)
    tc_target = TorchProbabilityCategoricalDistribution(torch.as_tensor(target_probs))
    assert coverage(np_cs, np_target) == pytest.approx(coverage(tc_cs, tc_target))
    assert efficiency(np_cs) == pytest.approx(efficiency(tc_cs))


def test_probability_intervals_numpy_torch_parity() -> None:
    lower = np.array([[0.1, 0.4, 0.05], [0.2, 0.2, 0.2]])
    upper = np.array([[0.5, 0.6, 0.2], [0.4, 0.4, 0.4]])
    target_probs = np.array([[0.2, 0.5, 0.1], [0.3, 0.3, 0.4]])
    np_cs = ArrayProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    tc_cs = TorchProbabilityIntervalsCredalSet(
        lower_bounds=torch.as_tensor(lower),
        upper_bounds=torch.as_tensor(upper),
    )
    np_target = ArrayProbabilityCategoricalDistribution(target_probs)
    tc_target = TorchProbabilityCategoricalDistribution(torch.as_tensor(target_probs))
    assert coverage(np_cs, np_target) == pytest.approx(coverage(tc_cs, tc_target))
    assert efficiency(np_cs) == pytest.approx(efficiency(tc_cs))


def test_unregistered_torch_credal_types_raise() -> None:
    """DistanceBased and DirichletLevelSet credal sets are intentionally not registered."""
    distance_cs = TorchDistanceBasedCredalSet(
        nominal=TorchProbabilityCategoricalDistribution(torch.tensor([[0.5, 0.3, 0.2]])),
        radius=torch.tensor([0.1]),
    )
    with pytest.raises(NotImplementedError, match="coverage is not implemented"):
        coverage(
            distance_cs,
            TorchProbabilityCategoricalDistribution(torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)),
        )
    with pytest.raises(NotImplementedError, match="efficiency is not implemented"):
        efficiency(distance_cs)

    dirichlet_cs = TorchDirichletLevelSetCredalSet(
        alphas=torch.tensor([[5.0, 5.0, 5.0]]),
        threshold=torch.tensor(0.5),
    )
    with pytest.raises(NotImplementedError, match="coverage is not implemented"):
        coverage(
            dirichlet_cs,
            TorchProbabilityCategoricalDistribution(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)),
        )
    with pytest.raises(NotImplementedError, match="efficiency is not implemented"):
        efficiency(dirichlet_cs)
