"""JAX credal-set evaluation tests for ``coverage`` / ``efficiency``.

Runs the shared :class:`CredalSuite` against the jax credal-set wrappers and
additionally asserts numerical parity with the numpy backend on identical
inputs. The Dirichlet-level-set jax handler is exercised on a hand-built
Dirichlet whose level set degenerates to (a small subset of) the simplex.

Torch is deliberately never imported here: the CI jax job installs no torch.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from probly.metrics import average_interval_width, convex_hull_coverage, coverage, efficiency
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
)
from probly.representation.credal_set.jax import (
    JaxConvexCredalSet,
    JaxDirichletLevelSetCredalSet,
    JaxDistanceBasedCredalSet,
    JaxProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.array_categorical import ArrayProbabilityCategoricalDistribution
from probly.representation.distribution.jax_categorical import JaxProbabilityCategoricalDistribution

from ._credal_suite import CredalSuite


@pytest.fixture
def array_fn():
    return jnp.asarray


@pytest.fixture
def make_convex():
    return lambda probs: JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(jnp.asarray(probs)))


@pytest.fixture
def make_distance():
    return lambda nominal, radius: JaxDistanceBasedCredalSet(
        nominal=JaxProbabilityCategoricalDistribution(jnp.asarray(nominal)),
        radius=jnp.asarray(radius),
    )


@pytest.fixture
def make_intervals():
    return lambda lower, upper: JaxProbabilityIntervalsCredalSet(
        lower_bounds=jnp.asarray(lower),
        upper_bounds=jnp.asarray(upper),
    )


class TestJax(CredalSuite):
    """JAX implementation of the shared credal suite."""


@pytest.mark.parametrize(
    "probs",
    [
        np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]]),
        np.array([[[0.7, 0.1, 0.1, 0.1], [0.5, 0.3, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]]]),
    ],
)
def test_convex_numpy_jax_parity(probs: np.ndarray) -> None:
    """Convex coverage and efficiency agree across backends on identical inputs."""
    np_cs = ArrayConvexCredalSet(array=ArrayProbabilityCategoricalDistribution(probs))
    jx_cs = JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(jnp.asarray(probs)))
    y = np.array([1])
    assert coverage(np_cs, y) == pytest.approx(coverage(jx_cs, jnp.asarray(y)))
    assert efficiency(np_cs) == pytest.approx(efficiency(jx_cs), abs=1e-6)


def test_distance_numpy_jax_parity() -> None:
    nominal = np.array([[0.5, 0.3, 0.2]])
    radius = np.array([0.1])
    np_cs = ArrayDistanceBasedCredalSet(
        nominal=ArrayProbabilityCategoricalDistribution(nominal),
        radius=radius,
    )
    jx_cs = JaxDistanceBasedCredalSet(
        nominal=JaxProbabilityCategoricalDistribution(jnp.asarray(nominal)),
        radius=jnp.asarray(radius),
    )
    y = np.array([0])
    assert coverage(np_cs, y) == pytest.approx(coverage(jx_cs, jnp.asarray(y)))
    assert efficiency(np_cs) == pytest.approx(efficiency(jx_cs), abs=1e-6)
    assert average_interval_width(np_cs) == pytest.approx(average_interval_width(jx_cs), abs=1e-6)


def test_probability_intervals_numpy_jax_parity() -> None:
    lower = np.array([[0.1, 0.4, 0.05], [0.2, 0.2, 0.2]])
    upper = np.array([[0.5, 0.6, 0.2], [0.4, 0.4, 0.4]])
    np_cs = ArrayProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    jx_cs = JaxProbabilityIntervalsCredalSet(
        lower_bounds=jnp.asarray(lower),
        upper_bounds=jnp.asarray(upper),
    )
    y = np.array([0, 1])
    assert coverage(np_cs, y) == pytest.approx(coverage(jx_cs, jnp.asarray(y)))
    assert efficiency(np_cs) == pytest.approx(efficiency(jx_cs), abs=1e-6)
    assert average_interval_width(np_cs) == pytest.approx(average_interval_width(jx_cs), abs=1e-6)


def test_distance_first_order_target_uses_tv_membership() -> None:
    """A probability-vector target routes through the TV-ball membership rule."""
    cs = JaxDistanceBasedCredalSet(
        nominal=JaxProbabilityCategoricalDistribution(jnp.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])),
        radius=jnp.array([0.1, 0.1]),
    )
    # TV(nominal, [0.55, 0.3, 0.15]) = 0.05 <= 0.1 -> covered.
    # TV(nominal, [0.9, 0.05, 0.05]) = 0.4 > 0.1 -> not covered.
    targets = jnp.array([[0.55, 0.3, 0.15], [0.9, 0.05, 0.05]])
    assert coverage(cs, targets) == pytest.approx(0.5)


def test_dirichlet_level_set_dispatches() -> None:
    """The Dirichlet-level-set handler resolves to concrete, deterministic values.

    The MC sampling uses the fixed default PRNG key of
    ``JaxDirichletLevelSetCredalSet``, so the values are reproducible without
    any global seeding. If a future refactor silently breaks the dispatch they
    will diverge.
    """
    cs = JaxDirichletLevelSetCredalSet(
        alphas=jnp.array([[5.0, 5.0, 5.0]]),
        threshold=jnp.array(0.5),
    )
    assert coverage(cs, jnp.array([0])) == pytest.approx(1.0, abs=0.05)
    assert efficiency(cs) == pytest.approx(3.0, abs=0.05)
    assert average_interval_width(cs) == pytest.approx(0.31, abs=0.05)


def test_jax_handlers_accept_numpy_targets() -> None:
    """A jax credal set accepts a numpy ``y_true`` without an explicit conversion."""
    cs = JaxDistanceBasedCredalSet(
        nominal=JaxProbabilityCategoricalDistribution(jnp.array([[0.5, 0.3, 0.2]])),
        radius=jnp.array([0.1]),
    )
    assert coverage(cs, np.array([0])) == pytest.approx(1.0)


def test_convex_hull_coverage_jax() -> None:
    """Hull coverage on a jax convex credal set routes through the numpy LP solver."""
    cs = JaxConvexCredalSet(
        tensor=JaxProbabilityCategoricalDistribution(
            jnp.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 2)
        ),
    )
    targets = JaxProbabilityCategoricalDistribution(jnp.array([[0.3, 0.4, 0.3], [0.1, 0.1, 0.8]]))
    assert float(convex_hull_coverage(cs, targets)) == pytest.approx(1.0)


def test_unregistered_jax_type_raises() -> None:
    """An unregistered jax wrapper raises a meaningful NotImplementedError.

    Pins the deliberate gap that no ``JaxSingletonCredalSet`` /
    ``JaxDiscreteCredalSet`` exist: a bare ``JaxCategoricalDistribution`` must
    not silently match an unintended handler.
    """
    distribution = JaxProbabilityCategoricalDistribution(jnp.array([[0.5, 0.5]]))
    with pytest.raises(NotImplementedError, match="coverage is not implemented"):
        coverage(distribution, jnp.array([0]))
    with pytest.raises(NotImplementedError, match="efficiency is not implemented"):
        efficiency(distribution)


def test_lazy_dispatch_loads_jax_module_on_credal_set() -> None:
    """Calling coverage on a jax credal set must trigger lazy loading of probly.metrics.jax.

    Locks in the contract that ``JAX_ARRAY_LIKE`` matches the credal-set
    wrappers (which inherit from ``JaxLikeImplementation``) and triggers the
    lazy import. Tested in a fresh subprocess so that prior test imports do not
    satisfy the ``in sys.modules`` check trivially.
    """
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    program = (
        "import sys; "
        "import jax.numpy as jnp; "
        "from probly.metrics import coverage; "
        "from probly.representation.credal_set.jax import JaxProbabilityIntervalsCredalSet; "
        "cs = JaxProbabilityIntervalsCredalSet("
        "lower_bounds=jnp.array([[0.1, 0.4, 0.05]]), "
        "upper_bounds=jnp.array([[0.5, 0.6, 0.2]]),"
        "); "
        "assert 'probly.metrics.jax' not in sys.modules, 'preloaded'; "
        "cov = coverage(cs, jnp.array([0])); "
        "assert 'probly.metrics.jax' in sys.modules, 'not lazy-loaded'; "
        "assert cov == 1.0, cov"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
