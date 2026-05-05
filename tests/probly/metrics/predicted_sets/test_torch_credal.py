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
from probly.representation.distribution.array_categorical import ArrayProbabilityCategoricalDistribution  # noqa: E402
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
    np_cs = ArrayConvexCredalSet(array=ArrayProbabilityCategoricalDistribution(probs))
    tc_cs = TorchConvexCredalSet(tensor=TorchCategoricalDistribution(torch.as_tensor(probs)))
    y = np.array([1])
    assert coverage(np_cs, y) == pytest.approx(coverage(tc_cs, torch.as_tensor(y)))
    assert efficiency(np_cs) == pytest.approx(efficiency(tc_cs))


def test_distance_numpy_torch_parity() -> None:
    nominal = np.array([[0.5, 0.3, 0.2]])
    radius = np.array([0.1])
    np_cs = ArrayDistanceBasedCredalSet(
        nominal=ArrayProbabilityCategoricalDistribution(nominal),
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
    """The Dirichlet-level-set handler resolves to concrete pinned-seed values.

    The MC sampling is stochastic, but with a fixed torch seed and 10000
    samples (the default in ``TorchDirichletLevelSetCredalSet``) the values
    are reproducible. The expected numbers below were computed from the
    pinned seed; if a future refactor silently breaks the dispatch they
    will diverge.
    """
    alphas = torch.tensor([[5.0, 5.0, 5.0]])
    threshold = torch.tensor(0.5)

    def _make_cs() -> TorchDirichletLevelSetCredalSet:
        return TorchDirichletLevelSetCredalSet(alphas=alphas, threshold=threshold)

    torch.manual_seed(0)
    cov = coverage(_make_cs(), torch.tensor([0]))
    torch.manual_seed(0)
    eff = efficiency(_make_cs())
    torch.manual_seed(0)
    width = average_interval_width(_make_cs())

    assert cov == pytest.approx(1.0, abs=0.05)
    assert eff == pytest.approx(3.0, abs=0.05)
    assert width == pytest.approx(0.31, abs=0.05)


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


def test_unregistered_torch_type_raises() -> None:
    """An unregistered torch wrapper raises a meaningful NotImplementedError.

    Pins the deliberate gap that no ``TorchSingletonCredalSet`` /
    ``TorchDiscreteCredalSet`` exist: a bare ``TorchCategoricalDistribution``
    must not silently match an unintended handler.
    """
    distribution = TorchCategoricalDistribution(torch.tensor([[0.5, 0.5]]))
    with pytest.raises(NotImplementedError, match="coverage is not implemented"):
        coverage(distribution, torch.tensor([0]))
    with pytest.raises(NotImplementedError, match="efficiency is not implemented"):
        efficiency(distribution)


def test_lazy_dispatch_loads_torch_module_on_credal_set() -> None:
    """Calling coverage on a torch credal set must trigger lazy loading of probly.metrics.torch.

    Locks in the contract that ``TORCH_TENSOR_LIKE`` matches the credal-set
    wrappers (which inherit from ``TorchLikeImplementation``) and triggers
    the lazy import. Tested in a fresh subprocess so that prior test
    imports do not satisfy the ``in sys.modules`` check trivially.
    """
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    program = (
        "import sys; "
        "import torch; "
        "from probly.metrics import coverage; "
        "from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet; "
        "cs = TorchProbabilityIntervalsCredalSet("
        "lower_bounds=torch.tensor([[0.1, 0.4, 0.05]]), "
        "upper_bounds=torch.tensor([[0.5, 0.6, 0.2]]),"
        "); "
        "assert 'probly.metrics.torch' not in sys.modules, 'preloaded'; "
        "cov = coverage(cs, torch.tensor([0])); "
        "assert 'probly.metrics.torch' in sys.modules, 'not lazy-loaded'; "
        "assert cov == 1.0, cov"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
