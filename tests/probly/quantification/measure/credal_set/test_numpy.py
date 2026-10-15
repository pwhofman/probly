"""Tests for numpy credal set uncertainty measures."""

from __future__ import annotations

from itertools import chain, combinations, product

import numpy as np
import pytest
from scipy.optimize import linprog

from probly.quantification import quantify
from probly.quantification.measure.credal_set import generalized_hartley, lower_entropy, upper_entropy
from probly.representation.credal_set.numpy import (
    NumpyConvexCredalSet,
    NumpyDistanceBasedCredalSet,
    NumpyProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.numpy_categorical import NumpyProbabilityCategoricalDistribution

from ._entropy_suite import (
    DTYPES,
    IntervalLowerEntropySuite,
    assert_in_intervals,
    entropy,
    polytope_vertices,
    random_intervals,
    tolerance,
)


class _NumpyBackend:
    """Builds numpy credal sets from numpy arrays for the shared correctness suite."""

    @staticmethod
    def intervals(lower: np.ndarray, upper: np.ndarray) -> NumpyProbabilityIntervalsCredalSet:
        return NumpyProbabilityIntervalsCredalSet(lower_bounds=np.asarray(lower), upper_bounds=np.asarray(upper))

    @staticmethod
    def numpy(value: np.ndarray) -> np.ndarray:
        return np.asarray(value)


@pytest.fixture
def backend() -> _NumpyBackend:
    return _NumpyBackend()


def _distance(nominal: np.ndarray, radius: np.ndarray | float) -> NumpyDistanceBasedCredalSet:
    return NumpyDistanceBasedCredalSet(
        nominal=NumpyProbabilityCategoricalDistribution(np.asarray(nominal)), radius=np.asarray(radius)
    )


def _convex(vertices: np.ndarray) -> NumpyConvexCredalSet:
    return NumpyConvexCredalSet(NumpyProbabilityCategoricalDistribution(np.asarray(vertices)))


def _gradient(p: np.ndarray) -> np.ndarray:
    """Entropy gradient ``-log(p) - 1``, with the log of exact zeros taken at the smallest positive float."""
    return -np.log(np.maximum(np.asarray(p, dtype=np.float64), np.finfo(np.float64).tiny)) - 1.0


# ---------------------------------------------------------------------------
# Probability intervals
# ---------------------------------------------------------------------------


class TestNumpyIntervalLowerEntropy(IntervalLowerEntropySuite):
    """Probability-interval lower entropy on numpy arrays."""


@DTYPES
@pytest.mark.parametrize("n_classes", [2, 3, 5, 10, 30])
def test_intervals_upper_entropy_is_the_maximum(dtype: type[np.floating], n_classes: int) -> None:
    """No point of the polytope improves the linear bound ``H(q) <= H(p) + g . (q - p)`` at the maximizer."""
    rng = np.random.default_rng(n_classes)
    bounds = [random_intervals(rng, kind, n_classes) for kind in ["ensemble", "dirichlet", "sparse", "box"] * 2]
    lower = np.stack([b[0] for b in bounds]).astype(dtype)
    upper = np.stack([b[1] for b in bounds]).astype(dtype)

    value, p = upper_entropy(_NumpyBackend.intervals(lower, upper), return_distribution=True)

    assert value.dtype == p.dtype == dtype
    assert_in_intervals(p, lower, upper, atol=tolerance(dtype))
    np.testing.assert_allclose(entropy(p), value, atol=tolerance(dtype))
    certificate = 1e-9 if dtype == np.float64 else 1e-4
    for row in range(len(bounds)):
        gradient = _gradient(p[row])
        result = linprog(
            -gradient, A_eq=np.ones((1, n_classes)), b_eq=[1.0], bounds=list(zip(lower[row], upper[row], strict=True))
        )
        assert -result.fun - gradient @ p[row] <= certificate, row


def test_intervals_upper_entropy_known_values() -> None:
    """A singleton gives its own entropy, the full simplex the uniform distribution."""
    point = np.array([0.2, 0.5, 0.3])
    credal_set = _NumpyBackend.intervals(np.stack([point, np.zeros(3)]), np.stack([point, np.ones(3)]))

    value, p = upper_entropy(credal_set, return_distribution=True)

    np.testing.assert_allclose(value, [entropy(point), np.log(3)], atol=1e-12)
    np.testing.assert_allclose(p, [point, np.full(3, 1 / 3)], atol=1e-12)
    np.testing.assert_allclose(upper_entropy(credal_set, base="normalize"), value / np.log(3))
    np.testing.assert_allclose(upper_entropy(credal_set, base=2.0), value / np.log(2))


def test_intervals_quantify_decomposes_into_upper_and_lower_entropy() -> None:
    """``quantify`` on numpy probability intervals gives total = upper and aleatoric = lower entropy."""
    rng = np.random.default_rng(0)
    bounds = [random_intervals(rng, "ensemble", 4) for _ in range(5)]
    credal_set = _NumpyBackend.intervals(np.stack([b[0] for b in bounds]), np.stack([b[1] for b in bounds]))

    decomposition = quantify(credal_set)

    np.testing.assert_allclose(decomposition.total, upper_entropy(credal_set))
    np.testing.assert_allclose(decomposition.aleatoric, lower_entropy(credal_set))
    np.testing.assert_allclose(decomposition.epistemic, decomposition.total - decomposition.aleatoric)
    assert np.all(decomposition.epistemic >= 0.0)


# ---------------------------------------------------------------------------
# Distance-based credal sets (total variation balls)
# ---------------------------------------------------------------------------


def _tv_ball_vertices(nominal: np.ndarray, radius: float) -> np.ndarray:
    """Vertices of ``{p : p >= 0, sum(p) = 1, s . (p - nominal) <= 2 radius for every sign vector s}``."""
    nominal = np.asarray(nominal, dtype=np.float64)
    n = nominal.shape[-1]
    signs = np.array(list(product([-1.0, 1.0], repeat=n)))
    return polytope_vertices(
        np.concatenate([-np.eye(n), signs]),
        np.concatenate([np.zeros(n), 2 * float(radius) + signs @ nominal]),
        np.ones((1, n)),
        np.ones(1),
    )


def _tv_ball_ascent_gap(p: np.ndarray, nominal: np.ndarray, radius: float) -> float:
    """Upper bound on ``max_{q in ball} H(q) - H(p)``, by a linear program over ``q = nominal + a - b``."""
    nominal = np.asarray(nominal, dtype=np.float64)
    n = nominal.shape[-1]
    gradient = _gradient(p)
    result = linprog(
        np.concatenate([-gradient, gradient]),
        A_ub=np.vstack([np.concatenate([np.ones(n), np.zeros(n)]), np.hstack([-np.eye(n), np.eye(n)])]),
        b_ub=np.concatenate([[float(radius)], nominal]),
        A_eq=np.concatenate([np.ones(n), -np.ones(n)])[None],
        b_eq=[0.0],
        bounds=[(0.0, None)] * (2 * n),
    )
    assert result.status == 0, result.message
    return float(gradient @ nominal - result.fun - gradient @ np.asarray(p, dtype=np.float64))


def _random_tv_balls(rng: np.random.Generator, n_sets: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Softmax, Dirichlet and sparse nominal distributions with radii in [0, 0.8)."""
    logits = rng.normal(size=(n_sets, n_classes)) * 2
    nominal = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    nominal[1::3] = rng.dirichlet(np.ones(n_classes), size=len(nominal[1::3]))
    sparse = rng.dirichlet(np.full(n_classes, 0.3), size=len(nominal[2::3]))
    sparse = np.where(sparse < 0.5 / n_classes, 0.0, sparse)
    sparse[sparse.sum(-1) == 0, 0] = 1.0
    nominal[2::3] = sparse / sparse.sum(-1, keepdims=True)
    return nominal, rng.uniform(0.0, 0.8, size=n_sets)


def _assert_in_tv_ball(p: np.ndarray, nominal: np.ndarray, radius: np.ndarray, atol: float) -> None:
    p = np.asarray(p, dtype=np.float64)
    assert (p >= -atol).all(), p
    np.testing.assert_allclose(p.sum(-1), 1.0, atol=atol)
    distance = 0.5 * np.abs(p - np.asarray(nominal, dtype=np.float64)).sum(-1)
    assert (distance - np.asarray(radius, dtype=np.float64) <= atol).all()


@DTYPES
def test_distance_regressions_against_the_per_class_box(dtype: type[np.floating]) -> None:
    """The per-class box is larger than the ball for more than three classes and gave ln 4 and ln 2 here."""
    upper_value, upper_p = upper_entropy(
        _distance(np.array([0.5, 0.5, 0.0, 0.0], dtype=dtype), np.array(0.3, dtype=dtype)), return_distribution=True
    )
    lower_value, lower_p = lower_entropy(
        _distance(np.full(4, 0.25, dtype=dtype), np.array(0.25, dtype=dtype)), return_distribution=True
    )

    np.testing.assert_allclose(upper_value, -2 * (0.35 * np.log(0.35) + 0.15 * np.log(0.15)), atol=tolerance(dtype))
    np.testing.assert_allclose(upper_p, [0.35, 0.35, 0.15, 0.15], atol=tolerance(dtype))
    np.testing.assert_allclose(lower_value, 1.5 * np.log(2), atol=tolerance(dtype))
    np.testing.assert_allclose(np.sort(lower_p), [0.0, 0.25, 0.25, 0.5], atol=tolerance(dtype))


@DTYPES
@pytest.mark.parametrize("n_classes", [2, 3, 4, 5])
def test_distance_entropies_match_vertex_enumeration(dtype: type[np.floating], n_classes: int) -> None:
    """The lower entropy is the minimum over the vertices of the ball; no vertex has more entropy than the upper."""
    nominal, radius = _random_tv_balls(np.random.default_rng(n_classes), 9, n_classes)
    credal_set = _distance(nominal.astype(dtype), radius.astype(dtype))

    upper_value, upper_p = upper_entropy(credal_set, return_distribution=True)
    lower_value, lower_p = lower_entropy(credal_set, return_distribution=True)

    vertex_entropies = [entropy(_tv_ball_vertices(c, r)) for c, r in zip(nominal, radius, strict=True)]
    np.testing.assert_allclose(lower_value, [h.min() for h in vertex_entropies], atol=tolerance(dtype))
    assert np.all(upper_value >= np.array([h.max() for h in vertex_entropies]) - tolerance(dtype))
    _assert_in_tv_ball(upper_p, nominal, radius, atol=tolerance(dtype))
    _assert_in_tv_ball(lower_p, nominal, radius, atol=tolerance(dtype))
    np.testing.assert_allclose(entropy(lower_p), lower_value, atol=tolerance(dtype))


@DTYPES
@pytest.mark.parametrize("n_classes", [2, 4, 10, 50])
def test_distance_upper_entropy_is_the_maximum(dtype: type[np.floating], n_classes: int) -> None:
    """The maximizer lies in the ball, and no point of the ball improves its linear bound."""
    nominal, radius = _random_tv_balls(np.random.default_rng(10 + n_classes), 9, n_classes)

    value, p = upper_entropy(_distance(nominal.astype(dtype), radius.astype(dtype)), return_distribution=True)

    assert value.dtype == p.dtype == dtype
    _assert_in_tv_ball(p, nominal, radius, atol=tolerance(dtype))
    np.testing.assert_allclose(entropy(p), value, atol=tolerance(dtype))
    certificate = 1e-9 if dtype == np.float64 else 1e-4
    for row in range(len(nominal)):
        assert _tv_ball_ascent_gap(p[row], nominal[row], radius[row]) <= certificate, row


def test_distance_edge_cases() -> None:
    """Radius zero or negative, radii that reach the uniform distribution or a corner, one and two classes."""
    nominal = np.array([[0.5, 0.3, 0.2, 0.0], [0.5, 0.3, 0.2, 0.0], [0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]])
    credal_set = _distance(nominal, np.array([0.0, -0.5, 0.2, 0.6]))

    upper_value = upper_entropy(credal_set)
    lower_value = lower_entropy(credal_set)

    expected_upper = [entropy(nominal[0]), entropy(nominal[0]), np.log(4), np.log(4)]
    np.testing.assert_allclose(upper_value, expected_upper, atol=1e-12)
    expected_lower = entropy(np.array([nominal[0], nominal[0], [0.6, 0.3, 0.1, 0.0], [1.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(lower_value, expected_lower, atol=1e-12)
    one_class = _distance(np.array([1.0]), 0.1)
    np.testing.assert_allclose([upper_entropy(one_class), lower_entropy(one_class)], 0.0, atol=1e-12)
    two_classes = _distance(np.array([0.7, 0.3]), 0.1)
    np.testing.assert_allclose(upper_entropy(two_classes), entropy(np.array([0.6, 0.4])))
    np.testing.assert_allclose(lower_entropy(two_classes), entropy(np.array([0.8, 0.2])))


def test_distance_batch_shapes_and_scalar_radius() -> None:
    """Multi-dimensional batches with per-set radii and a shared scalar radius agree row by row."""
    nominal, radius = _random_tv_balls(np.random.default_rng(5), 6, 6)

    for measure in (upper_entropy, lower_entropy):
        batched = measure(_distance(nominal.reshape(2, 3, 6), radius.reshape(2, 3)))
        shared = measure(_distance(nominal, radius[0]))
        assert batched.shape == (2, 3)
        np.testing.assert_allclose(
            batched.reshape(-1), [measure(_distance(c, r)) for c, r in zip(nominal, radius, strict=True)]
        )
        np.testing.assert_allclose(shared, [measure(_distance(c, radius[0])) for c in nominal])


def test_distance_lower_entropy_is_exact_for_every_approximate_value() -> None:
    credal_set = _distance(np.array([0.3, 0.2, 0.5]), 0.1)
    for approximate in (False, True, "auto"):
        np.testing.assert_allclose(lower_entropy(credal_set, approximate=approximate), lower_entropy(credal_set))


# ---------------------------------------------------------------------------
# Convex credal sets
# ---------------------------------------------------------------------------


def _random_vertices(rng: np.random.Generator, kind: str, n_vertices: int, n_classes: int) -> np.ndarray:
    """Vertices from an ensemble, Dirichlet draws, sparse draws with a class that is always zero, or repeated."""
    if kind == "ensemble":
        logits = rng.normal(size=n_classes) * 2 + rng.normal(size=(n_vertices, n_classes))
        return np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    if kind == "sparse":
        vertices = rng.dirichlet(np.full(n_classes, 0.3), size=n_vertices)
        vertices = np.where(vertices < 0.5 / n_classes, 0.0, vertices)
        vertices[:, -1] = 0.0
        vertices[vertices.sum(-1) == 0, 0] = 1.0
        return vertices / vertices.sum(-1, keepdims=True)
    vertices = rng.dirichlet(np.full(n_classes, 0.5), size=n_vertices)
    if kind == "repeated":
        vertices[-1] = vertices[0]
    return vertices


def _convex_ascent_gap(p: np.ndarray, vertices: np.ndarray) -> float:
    """Upper bound on ``max_{q in hull} H(q) - H(p)``; the linear bound is largest at a vertex."""
    gradient = _gradient(p)
    return float((np.asarray(vertices, dtype=np.float64) @ gradient).max() - np.asarray(p, dtype=np.float64) @ gradient)


def _convex_hull_distance(p: np.ndarray, vertices: np.ndarray) -> float:
    """L1 distance from ``p`` to the convex hull of ``vertices``, by a linear program."""
    vertices = np.asarray(vertices, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n, k = vertices.shape
    result = linprog(
        np.concatenate([np.zeros(n), np.ones(k)]),
        A_ub=np.block([[vertices.T, -np.eye(k)], [-vertices.T, -np.eye(k)]]),
        b_ub=np.concatenate([p, -p]),
        A_eq=np.concatenate([np.ones(n), np.zeros(k)])[None],
        b_eq=[1.0],
        bounds=[(0.0, None)] * (n + k),
    )
    assert result.status == 0, result.message
    return float(result.fun)


@DTYPES
@pytest.mark.parametrize(("n_vertices", "n_classes"), [(2, 3), (4, 3), (5, 10), (10, 10), (20, 5), (8, 50)])
def test_convex_upper_entropy_is_the_maximum(dtype: type[np.floating], n_vertices: int, n_classes: int) -> None:
    """The maximizer lies in the hull and no vertex improves its linear bound."""
    rng = np.random.default_rng(100 * n_vertices + n_classes)
    kinds = ["ensemble", "dirichlet", "sparse", "repeated"] * 2
    vertices = np.stack([_random_vertices(rng, kind, n_vertices, n_classes) for kind in kinds]).astype(dtype)

    value, p = upper_entropy(_convex(vertices), return_distribution=True)

    assert value.dtype == p.dtype == dtype
    np.testing.assert_allclose(entropy(p), value, atol=tolerance(dtype))
    certificate = 1e-9 if dtype == np.float64 else 1e-4
    for row in range(len(vertices)):
        assert _convex_hull_distance(p[row], vertices[row]) <= 10 * tolerance(dtype), row
        assert _convex_ascent_gap(p[row], vertices[row]) <= certificate, row


def test_convex_upper_entropy_edge_cases() -> None:
    """One class, repeated vertices, the corners of the simplex, a maximum at a vertex, a class always zero."""
    cases = [
        (np.ones((3, 1)), 0.0),
        (np.tile([0.2, 0.3, 0.5], (4, 1)), entropy(np.array([0.2, 0.3, 0.5]))),
        (np.eye(4), np.log(4)),
        (np.array([[1.0, 0.0], [0.8, 0.2]]), entropy(np.array([0.8, 0.2]))),
        (np.array([[0.9, 0.1, 0.0], [0.5, 0.5, 0.0]]), np.log(2)),
    ]
    for vertices, expected in cases:
        value, p = upper_entropy(_convex(vertices), return_distribution=True)
        np.testing.assert_allclose(value, expected, atol=1e-10)
        assert _convex_hull_distance(p, vertices) <= 1e-9


def test_convex_upper_entropy_does_not_depend_on_the_rest_of_the_batch() -> None:
    """A set gives the same value alone, in a large batch and in a multi-dimensional batch."""
    rng = np.random.default_rng(11)
    kinds = ["ensemble", "dirichlet", "sparse", "repeated"] * 50
    vertices = np.stack([_random_vertices(rng, kind, 10, 10) for kind in kinds])

    batched_value, batched_p = upper_entropy(_convex(vertices), return_distribution=True)
    nested = upper_entropy(_convex(vertices.reshape(10, 20, 10, 10)))

    np.testing.assert_allclose(nested.reshape(-1), batched_value, atol=1e-10)
    for row in [0, 1, 2, 3, 100, 199]:
        alone_value, alone_p = upper_entropy(_convex(vertices[row]), return_distribution=True)
        np.testing.assert_allclose(alone_value, batched_value[row], atol=1e-10)
        np.testing.assert_allclose(alone_p, batched_p[row], atol=1e-9)


def test_convex_upper_entropy_half_precision() -> None:
    """Half-precision sets are optimized in single precision and returned in half precision."""
    vertices = _random_vertices(np.random.default_rng(0), "dirichlet", 4, 3)

    value, p = upper_entropy(_convex(vertices.astype(np.float16)), return_distribution=True)

    assert value.dtype == p.dtype == np.float16
    np.testing.assert_allclose(value, upper_entropy(_convex(vertices)), atol=1e-2)


def test_convex_lower_entropy_is_the_smallest_vertex_entropy() -> None:
    vertices = np.stack([_random_vertices(np.random.default_rng(1), "dirichlet", 5, 4) for _ in range(3)])

    value, p = lower_entropy(_convex(vertices.reshape(3, 1, 5, 4)), return_distribution=True)

    assert value.shape == (3, 1)
    assert p.shape == (3, 1, 4)
    np.testing.assert_allclose(value[:, 0], entropy(vertices).min(-1))
    np.testing.assert_allclose(p[:, 0], vertices[np.arange(3), entropy(vertices).argmin(-1)])
    np.testing.assert_allclose(lower_entropy(_convex(vertices), base=2.0), entropy(vertices).min(-1) / np.log(2))
    for approximate in (False, True, "auto"):
        np.testing.assert_allclose(lower_entropy(_convex(vertices), approximate=approximate), value[:, 0])


def _moebius_generalized_hartley(vertices: np.ndarray) -> float:
    """Generalized Hartley measure by inclusion-exclusion over all subsets, in nats."""
    n_classes = vertices.shape[-1]

    def lower_probability(subset: tuple[int, ...]) -> float:
        return float(vertices[:, list(subset)].sum(-1).min()) if subset else 0.0

    result = 0.0
    for a in chain.from_iterable(combinations(range(n_classes), r) for r in range(1, n_classes + 1)):
        mass = sum(
            (-1) ** (len(a) - len(b)) * lower_probability(b) for r in range(len(a) + 1) for b in combinations(a, r)
        )
        result += mass * np.log(len(a))
    return result


@pytest.mark.parametrize("n_classes", [1, 2, 3, 5, 7])
def test_generalized_hartley_matches_inclusion_exclusion(n_classes: int) -> None:
    rng = np.random.default_rng(n_classes)
    vertices = np.stack([_random_vertices(rng, kind, 4, n_classes) for kind in ["ensemble", "dirichlet"] * 2])
    if n_classes == 1:
        vertices = np.ones((4, 4, 1))

    value = generalized_hartley(_convex(vertices))

    assert value.shape == (4,)
    np.testing.assert_allclose(value, [_moebius_generalized_hartley(v) for v in vertices], atol=1e-12)


def test_generalized_hartley_known_values() -> None:
    """A single distribution has none; the corners of the simplex have the Hartley measure of the class space."""
    np.testing.assert_allclose(generalized_hartley(_convex(np.array([[0.3, 0.5, 0.2]]))), 0.0, atol=1e-12)
    corners = _convex(np.eye(3))
    np.testing.assert_allclose(generalized_hartley(corners), np.log(3), atol=1e-12)
    np.testing.assert_allclose(generalized_hartley(corners, base=2.0), np.log2(3), atol=1e-12)
    np.testing.assert_allclose(generalized_hartley(corners, base="normalize"), 1.0, atol=1e-12)
