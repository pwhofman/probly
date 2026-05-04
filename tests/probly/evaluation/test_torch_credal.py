"""PyTorch credal-set evaluation tests for ``coverage`` / ``efficiency``.

Runs the shared :class:`CredalSuite` against the torch credal-set wrappers
and additionally asserts numerical parity with the numpy backend on identical
inputs. The Dirichlet-level-set torch handler is exercised on a hand-built
Dirichlet whose level set degenerates to (a small subset of) the simplex.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from probly.evaluation import average_interval_width, coverage, efficiency  # noqa: E402
from probly.representation.credal_set.array import (  # noqa: E402
    ArrayConvexCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
)
from probly.representation.credal_set.torch import (  # noqa: E402
    TorchConvexCredalSet,
    TorchDirichletLevelSetCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.representation.distribution import ArrayCategoricalDistribution  # noqa: E402
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: E402

from ._credal_suite import CredalSuite  # noqa: E402


@pytest.fixture
def array_fn():
    return torch.as_tensor


@pytest.fixture
def make_convex():
    return lambda probs: TorchConvexCredalSet(tensor=TorchCategoricalDistribution(torch.as_tensor(probs)))


@pytest.fixture
def make_distance():
    return lambda nominal, radius: TorchDistanceBasedCredalSet(
        nominal=TorchCategoricalDistribution(torch.as_tensor(nominal)),
        radius=torch.as_tensor(radius),
    )


@pytest.fixture
def make_intervals():
    return lambda lower, upper: TorchProbabilityIntervalsCredalSet(
        lower_bounds=torch.as_tensor(lower),
        upper_bounds=torch.as_tensor(upper),
    )


class TestTorch(CredalSuite):
    """PyTorch implementation of the shared credal suite."""


@pytest.mark.parametrize(
    "probs",
    [
        np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]]),
        np.array([[[0.7, 0.1, 0.1, 0.1], [0.5, 0.3, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]]]),
    ],
)
def test_convex_numpy_torch_parity(probs: np.ndarray) -> None:
    """Convex coverage and efficiency agree across backends on identical inputs."""
    np_cs = ArrayConvexCredalSet(array=ArrayCategoricalDistribution(probs))
    tc_cs = TorchConvexCredalSet(tensor=TorchCategoricalDistribution(torch.as_tensor(probs)))
    y = np.array([1])
    assert coverage(np_cs, y) == pytest.approx(coverage(tc_cs, torch.as_tensor(y)))
    assert efficiency(np_cs) == pytest.approx(efficiency(tc_cs))


def test_distance_numpy_torch_parity() -> None:
    nominal = np.array([[0.5, 0.3, 0.2]])
    radius = np.array([0.1])
    np_cs = ArrayDistanceBasedCredalSet(
        nominal=ArrayCategoricalDistribution(nominal),
        radius=radius,
    )
    tc_cs = TorchDistanceBasedCredalSet(
        nominal=TorchCategoricalDistribution(torch.as_tensor(nominal)),
        radius=torch.as_tensor(radius),
    )
    y = np.array([0])
    assert coverage(np_cs, y) == pytest.approx(coverage(tc_cs, torch.as_tensor(y)))
    assert efficiency(np_cs) == pytest.approx(efficiency(tc_cs))
    assert average_interval_width(np_cs) == pytest.approx(average_interval_width(tc_cs))


def test_probability_intervals_numpy_torch_parity() -> None:
    lower = np.array([[0.1, 0.4, 0.05], [0.2, 0.2, 0.2]])
    upper = np.array([[0.5, 0.6, 0.2], [0.4, 0.4, 0.4]])
    np_cs = ArrayProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    tc_cs = TorchProbabilityIntervalsCredalSet(
        lower_bounds=torch.as_tensor(lower),
        upper_bounds=torch.as_tensor(upper),
    )
    y = np.array([0, 1])
    assert coverage(np_cs, y) == pytest.approx(coverage(tc_cs, torch.as_tensor(y)))
    assert efficiency(np_cs) == pytest.approx(efficiency(tc_cs))
    assert average_interval_width(np_cs) == pytest.approx(average_interval_width(tc_cs))


def test_dirichlet_level_set_dispatches() -> None:
    """The Dirichlet-level-set handler resolves and returns finite numbers.

    The MC sampling makes the values stochastic, so we only check finiteness
    and the basic ``[0, 1]`` / non-negative ranges. Pinning a torch seed for
    determinism.
    """
    torch.manual_seed(0)
    alphas = torch.tensor([[5.0, 5.0, 5.0]])
    threshold = torch.tensor(0.5)
    cs = TorchDirichletLevelSetCredalSet(alphas=alphas, threshold=threshold)
    y = torch.tensor([0])
    cov = coverage(cs, y)
    eff = efficiency(cs)
    width = average_interval_width(cs)
    assert 0.0 <= cov <= 1.0
    assert eff >= 0.0
    assert width >= 0.0


def test_torch_handlers_respect_device() -> None:
    """A torch credal set with cpu data accepts a numpy ``y_true`` without device error."""
    nominal = torch.tensor([[0.5, 0.3, 0.2]])
    radius = torch.tensor([0.1])
    cs = TorchDistanceBasedCredalSet(
        nominal=TorchCategoricalDistribution(nominal),
        radius=radius,
    )
    # ``y_true`` is a numpy array; the handler must coerce on the right device.
    cov = coverage(cs, np.array([0]))
    assert cov == pytest.approx(1.0)
