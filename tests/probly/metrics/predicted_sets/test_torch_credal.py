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

from probly.metrics import average_interval_width, coverage, efficiency  # noqa: E402
from probly.representation.credal_set.numpy import (  # noqa: E402
    NumpyConvexCredalSet,
    NumpyDistanceBasedCredalSet,
    NumpyProbabilityIntervalsCredalSet,
)
from probly.representation.credal_set.torch import (  # noqa: E402
    TorchConvexCredalSet,
    TorchDirichletLevelSetCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.numpy_categorical import NumpyProbabilityCategoricalDistribution  # noqa: E402
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution  # noqa: E402

from ._credal_suite import CredalSuite  # noqa: E402


@pytest.fixture
def array_fn():
    return torch.as_tensor


@pytest.fixture
def make_convex():
    return lambda probs: TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(torch.as_tensor(probs)))


@pytest.fixture
def make_distance():
    return lambda nominal, radius: TorchDistanceBasedCredalSet(
        nominal=TorchProbabilityCategoricalDistribution(torch.as_tensor(nominal)),
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


@pytest.mark.parametrize("epsilon", [0.0, 0.0005, 0.005])
def test_envelope_coverage_dominates_hull_coverage_at_same_epsilon(epsilon: float) -> None:
    """The per-class envelope contains the convex hull, so at equal tolerance its coverage is never lower."""
    pytest.importorskip("scipy")
    from probly.metrics import convex_hull_coverage  # noqa: PLC0415

    gen = torch.Generator().manual_seed(0)
    vertices = torch.softmax(3.0 * torch.randn(200, 5, 4, generator=gen, dtype=torch.float64), dim=-1)
    targets = torch.softmax(3.0 * torch.randn(200, 4, generator=gen, dtype=torch.float64), dim=-1)
    # Half of the targets are mixtures of the vertices, so a share of them lies in or near the hull.
    weights = torch.softmax(torch.randn(100, 5, generator=gen, dtype=torch.float64), dim=-1)
    targets[:100] = torch.einsum("nv,nvc->nc", weights, vertices[:100])
    targets[:100] = targets[:100] + 1e-4 * torch.randn(100, 4, generator=gen, dtype=torch.float64)
    targets = targets.clamp_min(0.0) / targets.clamp_min(0.0).sum(-1, keepdim=True)

    cs = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))
    hull = convex_hull_coverage(cs, TorchProbabilityCategoricalDistribution(targets), epsilon=epsilon)
    envelope = coverage(cs, targets, epsilon=epsilon)
    assert envelope >= float(hull)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("delta", [0.0003, 0.0007, 0.0011, 0.002])
def test_probability_vector_coverage_boundary_is_precision_independent(dtype: torch.dtype, delta: float) -> None:
    """A target at L1 distance exactly ``epsilon`` is covered for single- and double-precision bounds alike."""
    cs = TorchProbabilityIntervalsCredalSet(
        lower_bounds=torch.tensor([[delta, 0.0, 0.0]], dtype=dtype),
        upper_bounds=torch.tensor([[0.5, 0.6, 0.6]], dtype=dtype),
    )
    target = torch.tensor([[0.0, 0.5, 0.5]], dtype=dtype)
    assert coverage(cs, target, epsilon=2 * delta) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [efficiency, average_interval_width])
def test_envelope_metrics_are_differentiable(metric) -> None:
    """Efficiency and interval width stay on the autograd graph of the credal set's probabilities."""
    logits = torch.randn(8, 5, 3, generator=torch.Generator().manual_seed(0), requires_grad=True)
    cs = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(torch.softmax(logits, dim=-1)))
    (grad,) = torch.autograd.grad(metric(cs), logits)
    assert grad.abs().sum() > 0


def test_set_metrics_return_tensors() -> None:
    """Set metrics return zero-dimensional tensors in the precision of the credal set."""
    lower = torch.tensor([[0.1, 0.4, 0.05]], dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=lower + 0.2)
    results = [coverage(cs, torch.tensor([1])), efficiency(cs), average_interval_width(cs)]
    assert all(isinstance(result, torch.Tensor) and result.ndim == 0 for result in results)
    assert efficiency(cs).dtype == torch.float64


@pytest.mark.parametrize(
    "probs",
    [
        np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]]),
        np.array([[[0.7, 0.1, 0.1, 0.1], [0.5, 0.3, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]]]),
    ],
)
def test_convex_numpy_torch_parity(probs: np.ndarray) -> None:
    """Convex coverage and efficiency agree across backends on identical inputs."""
    np_cs = NumpyConvexCredalSet(array=NumpyProbabilityCategoricalDistribution(probs))
    tc_cs = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(torch.as_tensor(probs)))
    y = np.array([1])
    assert coverage(np_cs, y) == pytest.approx(coverage(tc_cs, torch.as_tensor(y)))
    assert efficiency(np_cs) == pytest.approx(efficiency(tc_cs))


def test_distance_numpy_torch_parity() -> None:
    nominal = np.array([[0.5, 0.3, 0.2]])
    radius = np.array([0.1])
    np_cs = NumpyDistanceBasedCredalSet(
        nominal=NumpyProbabilityCategoricalDistribution(nominal),
        radius=radius,
    )
    tc_cs = TorchDistanceBasedCredalSet(
        nominal=TorchProbabilityCategoricalDistribution(torch.as_tensor(nominal)),
        radius=torch.as_tensor(radius),
    )
    y = np.array([0])
    assert coverage(np_cs, y) == pytest.approx(coverage(tc_cs, torch.as_tensor(y)))
    assert efficiency(np_cs) == pytest.approx(efficiency(tc_cs))
    assert average_interval_width(np_cs) == pytest.approx(average_interval_width(tc_cs))


def test_probability_intervals_numpy_torch_parity() -> None:
    lower = np.array([[0.1, 0.4, 0.05], [0.2, 0.2, 0.2]])
    upper = np.array([[0.5, 0.6, 0.2], [0.4, 0.4, 0.4]])
    np_cs = NumpyProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
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
        nominal=TorchProbabilityCategoricalDistribution(nominal),
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
    distribution = TorchProbabilityCategoricalDistribution(torch.tensor([[0.5, 0.5]]))
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
