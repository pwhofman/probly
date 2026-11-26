"""Backend-agnostic correctness suite for the credal-set entropies.

Every expected value comes from an independent numpy reference implemented in
this module, never from the code under test:

* ``polytope_vertices`` enumerates the vertices of a polytope by brute force
  from its inequality description. It gives the exact lower entropy of
  probability-interval credal sets, since entropy is concave and attains its
  minimum over a polytope at a vertex.
* ``interval_min_entropy_by_free_class`` enumerates the same vertices faster, by
  free class, for the class counts where the brute force is too slow.
* ``previous_greedy_min_entropy`` is the greedy heuristic of the library, used with
  ``approximate``.
* ``tv_ball_vertices`` and ``tv_ball_candidates`` give the exact lower entropy
  of total-variation balls, the first from the ball's inequalities and the
  second from every choice of receiving and emptied classes.
  ``tv_ball_min_entropy_by_receiver`` gives it for many classes, from one box
  per receiving class.
* ``tv_ball_max_entropy`` solves the smooth lifted problem with SLSQP, and
  ``tv_ball_ascent_gap`` solves a linear program whose value bounds how far a
  point is below the maximum entropy over the ball.

Backend test modules subclass the suites and provide a ``backend`` fixture
that builds credal sets from numpy arrays and converts results back.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Protocol
import warnings

import numpy as np
import pytest
from scipy.optimize import linprog, minimize

from probly.quantification.measure.credal_set import lower_entropy, upper_entropy

if TYPE_CHECKING:
    from collections.abc import Callable

# Largest number of classes for which the library computes the exact lower entropy.
EXACT_MAX_CLASSES = 14

DTYPES = pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])


def tolerance(dtype: type[np.floating]) -> float:
    """Absolute tolerance for comparing entropies computed in ``dtype`` with float64 references."""
    return 1e-10 if dtype == np.float64 else 1e-5


class Backend(Protocol):
    """Builds backend credal sets from numpy arrays and converts backend arrays back."""

    def intervals(self, lower: np.ndarray, upper: np.ndarray) -> Any:  # noqa: ANN401
        """Build a probability-intervals credal set."""
        ...

    def distance(self, nominal: np.ndarray, radius: np.ndarray) -> Any:  # noqa: ANN401
        """Build a distance-based (total variation) credal set."""
        ...

    def numpy(self, value: Any) -> np.ndarray:  # noqa: ANN401
        """Convert a backend array to numpy."""
        ...


# ---------------------------------------------------------------------------
# Numpy references
# ---------------------------------------------------------------------------


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats along the last axis, with ``0 log 0 = 0``."""
    p = np.asarray(p, dtype=np.float64)
    safe = np.where(p > 0, p, 1.0)
    return -(p * np.log(safe)).sum(-1)


def polytope_vertices(
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    *,
    tol: float = 1e-9,
) -> np.ndarray:
    """Enumerate the vertices of ``{x : a_ub x <= b_ub, a_eq x = b_eq}`` by brute force.

    A vertex is a feasible point at which ``dim`` linearly independent constraints are
    active. Every choice of ``dim - len(b_eq)`` inequalities is solved together with the
    equalities, and the feasible solutions of the nonsingular systems are the vertices.
    """
    dim = a_ub.shape[1]
    n_active = dim - a_eq.shape[0]
    combinations = list(itertools.combinations(range(a_ub.shape[0]), n_active))
    rows = np.array(combinations, dtype=int).reshape(len(combinations), n_active)
    found = []
    for chunk in np.array_split(rows, max(1, len(rows) // 20000)):
        a = np.concatenate([np.broadcast_to(a_eq, (len(chunk), *a_eq.shape)), a_ub[chunk]], axis=1)
        b = np.concatenate([np.broadcast_to(b_eq, (len(chunk), len(b_eq))), b_ub[chunk]], axis=1)
        # The constraint normals are integer vectors here, so a nonsingular system has |det| >= 1.
        regular = np.abs(np.linalg.det(a)) > 1e-6
        x = np.linalg.solve(a[regular], b[regular][..., None])[..., 0]
        feasible = (x @ a_ub.T <= b_ub + tol).all(-1) & (np.abs(x @ a_eq.T - b_eq) <= tol).all(-1)
        found.append(x[feasible])
    return np.concatenate(found)


def interval_vertices(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Vertices of ``{p : lower <= p <= upper, sum(p) = 1}``."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    n = lower.shape[-1]
    eye = np.eye(n)
    return polytope_vertices(
        np.concatenate([eye, -eye]),
        np.concatenate([upper, -lower]),
        np.ones((1, n)),
        np.ones(1),
    )


def interval_min_entropy(lower: np.ndarray, upper: np.ndarray) -> float:
    """Exact lower entropy of a probability-intervals credal set by vertex enumeration."""
    return float(entropy(interval_vertices(lower, upper)).min())


def interval_min_entropy_by_free_class(lower: np.ndarray, upper: np.ndarray) -> float:
    """Exact lower entropy of a probability-intervals credal set for moderate ``n``.

    Every vertex has all classes but one (the free class ``f``) at a bound. For each ``f``
    and each subset of the other classes at their upper bound, the free class takes the
    remaining mass, and the point is a vertex if that mass is within the bounds of ``f``.
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    n = lower.shape[-1]
    subsets = ((np.arange(2 ** (n - 1))[:, None] >> np.arange(n - 1)) & 1).astype(bool)
    best = np.inf
    for free in range(n):
        others = np.delete(np.arange(n), free)
        p = np.broadcast_to(lower, (len(subsets), n)).copy()
        p[:, others] = np.where(subsets, upper[others], lower[others])
        p[:, free] = 1.0 - p[:, others].sum(1)
        ok = (p[:, free] >= lower[free] - 1e-12) & (p[:, free] <= upper[free] + 1e-12)
        if ok.any():
            best = min(best, float(entropy(np.clip(p[ok], 0.0, None)).min()))
    return best


def previous_greedy_min_entropy(lower: np.ndarray, upper: np.ndarray) -> float:
    """The greedy heuristic: fill class j first, then the rest in index order, for every j."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    n = lower.shape[-1]
    best = np.inf
    for first in range(n):
        p = lower.copy()
        remaining = 1.0 - lower.sum()
        for i in [first, *(k for k in range(n) if k != first)]:
            fill = min(max(remaining, 0.0), upper[i] - lower[i])
            p[i] += fill
            remaining -= fill
        best = min(best, float(entropy(p)))
    return best


def random_intervals(rng: np.random.Generator, kind: str, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw a feasible probability-interval credal set of a given kind.

    Kinds:
        ``ensemble``: envelope of five softmax outputs, as built by the credal wrapper.
        ``dirichlet``: envelope of five Dirichlet(0.7) draws, where a greedy search often misses the minimum.
        ``sparse``: envelope of Dirichlet(0.1) draws, with lower bounds that are exactly zero.
        ``box``: a distribution widened by random amounts and clipped to [0, 1].
    """
    if kind == "ensemble":
        logits = rng.normal(size=n_classes) * 2 + rng.normal(size=(5, n_classes)) * 0.7
        members = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    elif kind == "dirichlet":
        members = rng.dirichlet(np.full(n_classes, 0.7), size=5)
    elif kind == "sparse":
        members = rng.dirichlet(np.full(n_classes, 0.1), size=4)
        members = np.where(members < 1e-3, 0.0, members)
        members /= members.sum(-1, keepdims=True)
    elif kind == "box":
        center = rng.dirichlet(np.ones(n_classes))
        lower = np.clip(center - rng.uniform(0.0, 0.2, n_classes), 0.0, 1.0)
        upper = np.clip(center + rng.uniform(0.0, 0.2, n_classes), 0.0, 1.0)
        return lower, upper
    else:
        msg = f"Unknown kind {kind}."
        raise ValueError(msg)
    return members.min(0), members.max(0)


def total_variation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total variation distance along the last axis."""
    return 0.5 * np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum(-1)


def tv_ball_vertices(nominal: np.ndarray, radius: float) -> np.ndarray:
    """Vertices of ``{p : p >= 0, sum(p) = 1, TV(p, nominal) <= radius}``.

    The total variation constraint is ``s . (p - nominal) <= 2 radius`` for every sign
    vector ``s``, so there are ``2**n`` inequalities; use this for at most five classes.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    n = nominal.shape[-1]
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    return polytope_vertices(
        np.concatenate([-np.eye(n), signs]),
        np.concatenate([np.zeros(n), 2 * float(radius) + signs @ nominal]),
        np.ones((1, n)),
        np.ones(1),
    )


def tv_ball_candidates(nominal: np.ndarray, radius: float) -> np.ndarray:
    """Points of the total-variation ball that include all of its vertices.

    At a vertex of the ball at most one class lies above the nominal (the receiver) and
    at most one class lies strictly between zero and its nominal value, since mass could
    otherwise move between two such classes in either direction without leaving the ball.
    So every vertex empties some set of classes into one receiver and may also take the
    rest of the budget from one more class. This enumerates all such points.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    radius = float(radius)
    n = nominal.shape[-1]
    points = [nominal.copy()]
    for receiver in range(n):
        others = [i for i in range(n) if i != receiver]
        for size in range(len(others) + 1):
            for emptied in itertools.combinations(others, size):
                mass = nominal[list(emptied)].sum()
                if mass > radius:
                    continue
                p = nominal.copy()
                p[list(emptied)] = 0.0
                p[receiver] += mass
                points.append(p)
                rest = radius - mass
                for donor in others:
                    if donor not in emptied and nominal[donor] > rest:
                        q = p.copy()
                        q[donor] -= rest
                        q[receiver] += rest
                        points.append(q)
    return np.array(points)


def tv_ball_min_entropy(nominal: np.ndarray, radius: float) -> float:
    """Exact lower entropy of a total-variation ball."""
    return float(entropy(tv_ball_candidates(nominal, radius)).min())


def tv_ball_min_entropy_by_receiver(nominal: np.ndarray, radius: float) -> float:
    """Exact lower entropy of a total-variation ball for any number of classes.

    A vertex of the ball has at most one class above the nominal, the receiver ``j``, so the
    minimum over the ball is the smallest minimum over the sets ``{p : 0 <= p <= u, sum(p) = 1}``
    with ``u = nominal`` except ``u_j = nominal_j + radius``. With zero lower bounds, filling the
    classes in decreasing order of ``u`` gives a point that majorizes every other point of such
    a set, so no point of the set has less entropy.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    best = np.inf
    for receiver in range(nominal.shape[-1]):
        upper = nominal.copy()
        upper[receiver] += float(radius)
        upper = np.sort(upper)[::-1]
        filled = np.clip(nominal.sum() - np.concatenate([[0.0], np.cumsum(upper)[:-1]]), 0.0, upper)
        best = min(best, float(entropy(filled)))
    return best


def tv_ball_max_entropy(nominal: np.ndarray, radius: float) -> float:
    """Maximum entropy over a total-variation ball by SLSQP.

    The ball is written with the mass added, ``a``, and removed, ``b``: ``p = nominal + a - b``
    with ``a, b >= 0``, ``sum(a) = sum(b) <= radius`` and ``p >= 0``, which makes every
    constraint smooth. Several starting points guard against local stalls.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    radius = float(radius)
    n = nominal.shape[-1]

    def point(x: np.ndarray) -> np.ndarray:
        return nominal + x[:n] - x[n:]

    constraints = [
        {"type": "eq", "fun": lambda x: x[:n].sum() - x[n:].sum()},
        {"type": "ineq", "fun": lambda x: radius - x[:n].sum()},
        {"type": "ineq", "fun": point},
    ]
    starts = [np.zeros(2 * n), np.concatenate([np.full(n, radius / n), np.minimum(nominal, radius / n)])]
    best = -np.inf
    for start in starts:
        result = minimize(
            lambda x: -entropy(np.clip(point(x), 0.0, None)),
            start,
            method="SLSQP",
            bounds=[(0.0, None)] * (2 * n),
            constraints=constraints,
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        best = max(best, -float(result.fun))
    return best


def tv_ball_ascent_gap(p: np.ndarray, nominal: np.ndarray, radius: float) -> float:
    """Upper bound on ``max_{q in ball} H(q) - H(p)`` for a point ``p`` of the ball.

    Entropy is concave, so ``H(q) <= H(p) + g . (q - p)`` with ``g = -log(p) - 1``. The
    maximum of this linear bound over the ball is a linear program in the lifted variables
    of ``tv_ball_max_entropy``.
    """
    p = np.asarray(p, dtype=np.float64)
    nominal = np.asarray(nominal, dtype=np.float64)
    n = nominal.shape[-1]
    gradient = -np.log(np.maximum(p, np.finfo(np.float64).tiny)) - 1.0
    result = linprog(
        np.concatenate([-gradient, gradient]),
        A_ub=np.vstack([np.concatenate([np.ones(n), np.zeros(n)]), np.hstack([-np.eye(n), np.eye(n)])]),
        b_ub=np.concatenate([[float(radius)], nominal]),
        A_eq=np.concatenate([np.ones(n), -np.ones(n)])[None],
        b_eq=[0.0],
        bounds=[(0.0, None)] * (2 * n),
        method="highs",
    )
    assert result.status == 0, result.message
    return float(gradient @ nominal - result.fun - gradient @ p)


def random_tv_ball(rng: np.random.Generator, kind: str, n_classes: int) -> tuple[np.ndarray, float]:
    """Draw a nominal distribution of a given kind and a radius in [0, 0.8).

    Kinds:
        ``softmax``: softmax of scaled Gaussian logits, like a classifier output.
        ``dirichlet``: a Dirichlet(1) draw.
        ``sparse``: a Dirichlet(0.3) draw with its small entries set exactly to zero.
    """
    if kind == "softmax":
        logits = rng.normal(size=n_classes) * 2
        nominal = np.exp(logits) / np.exp(logits).sum()
    elif kind == "dirichlet":
        nominal = rng.dirichlet(np.ones(n_classes))
    elif kind == "sparse":
        nominal = rng.dirichlet(np.full(n_classes, 0.3))
        nominal = np.where(nominal < 0.5 / n_classes, 0.0, nominal)
        nominal = nominal / nominal.sum() if nominal.sum() > 0 else np.eye(n_classes)[0]
    else:
        msg = f"Unknown kind {kind}."
        raise ValueError(msg)
    return nominal, float(rng.uniform(0.0, 0.8))


def assert_in_intervals(p: np.ndarray, lower: np.ndarray, upper: np.ndarray, atol: float) -> None:
    """Check that ``p`` is a distribution inside the probability intervals."""
    p = np.asarray(p, dtype=np.float64)
    assert (p >= np.asarray(lower, dtype=np.float64) - atol).all(), (p, lower)
    assert (p <= np.asarray(upper, dtype=np.float64) + atol).all(), (p, upper)
    np.testing.assert_allclose(p.sum(-1), 1.0, atol=atol)


# ---------------------------------------------------------------------------
# Probability intervals (and the Dirichlet level sets that share their polytope)
# ---------------------------------------------------------------------------


_INTERVAL_KINDS = ["ensemble", "dirichlet", "sparse", "box"]


class IntervalLowerEntropySuite:
    """Lower entropy of probability-interval credal sets against brute-force references."""

    backend: Callable[..., Backend]

    @DTYPES
    def test_lower_entropy_reaches_the_vertex_the_greedy_search_missed(
        self, backend: Backend, dtype: type[np.floating]
    ) -> None:
        """Regression: the greedy search returned 0.9433 at (.4, .5, .1); the minimum is ln 2 at (0, .5, .5)."""
        lower = np.array([0.0, 0.1, 0.1], dtype=dtype)
        upper = np.array([0.4, 0.5, 0.5], dtype=dtype)

        value, p = lower_entropy(backend.intervals(lower, upper), return_distribution=True)

        np.testing.assert_allclose(backend.numpy(value), np.log(2), atol=tolerance(dtype))
        np.testing.assert_allclose(backend.numpy(p), [0.0, 0.5, 0.5], atol=tolerance(dtype))

    @DTYPES
    @pytest.mark.parametrize("n_classes", range(2, 9))
    @pytest.mark.parametrize("kind", _INTERVAL_KINDS)
    def test_lower_entropy_matches_vertex_enumeration(
        self, backend: Backend, dtype: type[np.floating], n_classes: int, kind: str
    ) -> None:
        """Below the threshold the search is exact: it equals the minimum over all vertices."""
        rng = np.random.default_rng(1000 * n_classes + _INTERVAL_KINDS.index(kind))
        bounds = [random_intervals(rng, kind, n_classes) for _ in range(6)]
        lower = np.stack([b[0] for b in bounds]).astype(dtype)
        upper = np.stack([b[1] for b in bounds]).astype(dtype)

        value, p = lower_entropy(backend.intervals(lower, upper), return_distribution=True)
        value, p = backend.numpy(value), backend.numpy(p)

        expected = [interval_min_entropy(lo, up) for lo, up in zip(lower, upper, strict=True)]
        np.testing.assert_allclose(value, expected, atol=tolerance(dtype))
        assert_in_intervals(p, lower, upper, atol=tolerance(dtype))
        np.testing.assert_allclose(entropy(p), value, atol=tolerance(dtype))

    @DTYPES
    @pytest.mark.parametrize("n_classes", [EXACT_MAX_CLASSES - 1, EXACT_MAX_CLASSES])
    def test_lower_entropy_is_exact_up_to_the_threshold(
        self, backend: Backend, dtype: type[np.floating], n_classes: int
    ) -> None:
        """The largest class counts that are still computed exactly."""
        rng = np.random.default_rng(n_classes)
        bounds = [random_intervals(rng, kind, n_classes) for kind in _INTERVAL_KINDS]
        lower = np.stack([b[0] for b in bounds]).astype(dtype)
        upper = np.stack([b[1] for b in bounds]).astype(dtype)

        value, p = lower_entropy(backend.intervals(lower, upper), return_distribution=True)
        value, p = backend.numpy(value), backend.numpy(p)

        expected = [interval_min_entropy_by_free_class(lo, up) for lo, up in zip(lower, upper, strict=True)]
        np.testing.assert_allclose(value, expected, atol=tolerance(dtype))
        assert_in_intervals(p, lower, upper, atol=tolerance(dtype))

    @DTYPES
    @pytest.mark.parametrize("n_classes", [3, EXACT_MAX_CLASSES + 2])
    @pytest.mark.parametrize("kind", _INTERVAL_KINDS)
    def test_lower_entropy_approximate_is_the_greedy_search(
        self, backend: Backend, dtype: type[np.floating], n_classes: int, kind: str
    ) -> None:
        """``approximate=True`` uses the greedy search for any number of classes, without a warning."""
        rng = np.random.default_rng(7 * n_classes + _INTERVAL_KINDS.index(kind))
        bounds = [random_intervals(rng, kind, n_classes) for _ in range(8)]
        lower = np.stack([b[0] for b in bounds]).astype(dtype)
        upper = np.stack([b[1] for b in bounds]).astype(dtype)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = backend.numpy(lower_entropy(backend.intervals(lower, upper), approximate=True))

        expected = [previous_greedy_min_entropy(lo, up) for lo, up in zip(lower, upper, strict=True)]
        np.testing.assert_allclose(value, expected, atol=tolerance(dtype))

    def test_lower_entropy_auto_is_exact_up_to_the_threshold(self, backend: Backend) -> None:
        """``approximate="auto"`` gives the exact value, without a warning, when that is feasible."""
        lower = np.array([0.0, 0.1, 0.1])
        upper = np.array([0.4, 0.5, 0.5])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = backend.numpy(lower_entropy(backend.intervals(lower, upper), approximate="auto"))

        np.testing.assert_allclose(value, np.log(2), atol=1e-10)

    @pytest.mark.parametrize("kwargs", [{}, {"approximate": "auto"}])
    def test_lower_entropy_auto_warns_and_approximates_above_the_threshold(
        self, backend: Backend, kwargs: dict
    ) -> None:
        """Above the threshold, ``approximate="auto"`` uses the greedy search and says so."""
        lower, upper = random_intervals(np.random.default_rng(5), "dirichlet", EXACT_MAX_CLASSES + 1)

        with pytest.warns(UserWarning, match="approximated by a greedy search"):
            value = backend.numpy(lower_entropy(backend.intervals(lower, upper), **kwargs))

        np.testing.assert_allclose(value, previous_greedy_min_entropy(lower, upper), atol=1e-10)

    def test_lower_entropy_exact_raises_above_the_threshold(self, backend: Backend) -> None:
        """An explicit request for exact computation raises when there are too many classes."""
        lower, upper = random_intervals(np.random.default_rng(5), "dirichlet", EXACT_MAX_CLASSES + 1)

        with pytest.raises(ValueError, match="approximate=True"):
            lower_entropy(backend.intervals(lower, upper), approximate=False)

    def test_lower_entropy_rejects_an_unknown_approximate(self, backend: Backend) -> None:
        with pytest.raises(ValueError, match="approximate must be"):
            lower_entropy(backend.intervals(np.array([0.2, 0.3]), np.array([0.7, 0.8])), approximate="yes")

    @DTYPES
    @pytest.mark.parametrize(("n_classes", "approximate"), [(4, False), (EXACT_MAX_CLASSES + 2, True)])
    def test_lower_entropy_degenerate_sets(
        self, backend: Backend, dtype: type[np.floating], n_classes: int, approximate: bool
    ) -> None:
        """Singletons, one-hot singletons, the full simplex and bounds that already sum to one."""
        rng = np.random.default_rng(0)
        point = rng.dirichlet(np.ones(n_classes))
        one_hot = np.eye(n_classes)[1]
        zeros, ones = np.zeros(n_classes), np.ones(n_classes)
        tight_lower = np.full(n_classes, 1.0 / n_classes)
        lower = np.stack([point, one_hot, zeros, tight_lower]).astype(dtype)
        upper = np.stack([point, one_hot, ones, tight_lower + 0.5]).astype(dtype)

        value, p = lower_entropy(backend.intervals(lower, upper), return_distribution=True, approximate=approximate)
        value, p = backend.numpy(value), backend.numpy(p)

        expected = np.array([float(entropy(point)), 0.0, 0.0, np.log(n_classes)])
        np.testing.assert_allclose(value, expected, atol=tolerance(dtype))
        np.testing.assert_allclose(p[0], point, atol=tolerance(dtype))
        np.testing.assert_allclose(p[1], one_hot, atol=tolerance(dtype))
        np.testing.assert_allclose(p[2].max(), 1.0, atol=tolerance(dtype))
        assert_in_intervals(p, lower, upper, atol=tolerance(dtype))

    @DTYPES
    def test_lower_entropy_single_class(self, backend: Backend, dtype: type[np.floating]) -> None:
        """With one class the only distribution is (1,), whatever the bounds allow."""
        lower = np.array([[1.0], [0.0]], dtype=dtype)
        upper = np.array([[1.0], [1.0]], dtype=dtype)

        value, p = lower_entropy(backend.intervals(lower, upper), return_distribution=True)

        np.testing.assert_allclose(backend.numpy(value), [0.0, 0.0], atol=tolerance(dtype))
        np.testing.assert_allclose(backend.numpy(p), [[1.0], [1.0]], atol=tolerance(dtype))

    @pytest.mark.parametrize(("n_classes", "approximate"), [(5, False), (EXACT_MAX_CLASSES + 5, True)])
    def test_lower_entropy_batch_shapes(self, backend: Backend, n_classes: int, approximate: bool) -> None:
        """Unbatched, batched and multi-dimensional batches give the same row-wise results."""
        rng = np.random.default_rng(3)
        bounds = [random_intervals(rng, "ensemble", n_classes) for _ in range(6)]
        lower = np.stack([b[0] for b in bounds]).reshape(2, 3, n_classes)
        upper = np.stack([b[1] for b in bounds]).reshape(2, 3, n_classes)

        batched = backend.numpy(lower_entropy(backend.intervals(lower, upper), approximate=approximate))
        single = [
            float(backend.numpy(lower_entropy(backend.intervals(lo, up), approximate=approximate)))
            for lo, up in zip(lower.reshape(6, -1), upper.reshape(6, -1), strict=True)
        ]

        assert batched.shape == (2, 3)
        np.testing.assert_allclose(batched.reshape(-1), single, atol=1e-12)

    @pytest.mark.parametrize(("n_classes", "approximate"), [(6, False), (EXACT_MAX_CLASSES + 3, True)])
    def test_lower_entropy_does_not_depend_on_the_rest_of_the_batch(
        self, backend: Backend, n_classes: int, approximate: bool
    ) -> None:
        """A set gives the same value alone and inside a large random batch."""
        rng = np.random.default_rng(4)
        bounds = [random_intervals(rng, "dirichlet", n_classes) for _ in range(300)]
        lower = np.stack([b[0] for b in bounds])
        upper = np.stack([b[1] for b in bounds])

        batched = backend.numpy(lower_entropy(backend.intervals(lower, upper), approximate=approximate))

        for row in [0, 123, 299]:
            alone = backend.numpy(
                lower_entropy(backend.intervals(lower[row : row + 1], upper[row : row + 1]), approximate=approximate)
            )
            np.testing.assert_allclose(alone, batched[row : row + 1], atol=1e-12)

    @DTYPES
    def test_lower_entropy_with_base(self, backend: Backend, dtype: type[np.floating]) -> None:
        """The log base rescales the value and leaves the minimizer unchanged."""
        lower = np.array([0.0, 0.1, 0.1, 0.05], dtype=dtype)
        upper = np.array([0.4, 0.5, 0.5, 0.3], dtype=dtype)
        credal_set = backend.intervals(lower, upper)

        natural, p_natural = lower_entropy(credal_set, return_distribution=True)
        base_two, p_base_two = lower_entropy(credal_set, base=2.0, return_distribution=True)
        normalized = lower_entropy(credal_set, base="normalize")

        expected = interval_min_entropy(lower, upper)
        np.testing.assert_allclose(backend.numpy(natural), expected, atol=tolerance(dtype))
        np.testing.assert_allclose(backend.numpy(base_two), expected / np.log(2), atol=tolerance(dtype))
        np.testing.assert_allclose(backend.numpy(normalized), expected / np.log(4), atol=tolerance(dtype))
        np.testing.assert_allclose(backend.numpy(p_natural), backend.numpy(p_base_two))


def assert_in_tv_ball(p: np.ndarray, nominal: np.ndarray, radius: np.ndarray, atol: float) -> None:
    """Check that ``p`` is a distribution within total variation ``radius`` of ``nominal``."""
    p = np.asarray(p, dtype=np.float64)
    assert (p >= -atol).all(), p
    np.testing.assert_allclose(p.sum(-1), 1.0, atol=atol)
    excess = total_variation(p, nominal) - np.asarray(radius, dtype=np.float64)
    assert (excess <= atol).all(), excess


_TV_KINDS = ["softmax", "dirichlet", "sparse"]


class DistanceBasedEntropySuite:
    """Upper and lower entropy of total-variation balls against independent references."""

    backend: Callable[..., Backend]

    @DTYPES
    def test_upper_entropy_moves_mass_from_the_largest_to_the_smallest_classes(
        self, backend: Backend, dtype: type[np.floating]
    ) -> None:
        """Regression: the box around (.5, .5, 0, 0) with radius .3 gave ln 4 at the uniform point (TV .5).

        Water-filling moves mass .3: the two large classes drop to .35 and the two empty
        ones rise to .15.
        """
        nominal = np.array([0.5, 0.5, 0.0, 0.0], dtype=dtype)

        value, p = upper_entropy(backend.distance(nominal, np.array(0.3, dtype=dtype)), return_distribution=True)

        expected = -2 * (0.35 * np.log(0.35) + 0.15 * np.log(0.15))
        np.testing.assert_allclose(backend.numpy(value), expected, atol=tolerance(dtype))
        np.testing.assert_allclose(backend.numpy(p), [0.35, 0.35, 0.15, 0.15], atol=tolerance(dtype))

    @DTYPES
    def test_lower_entropy_moves_the_radius_into_one_class(self, backend: Backend, dtype: type[np.floating]) -> None:
        """Regression: the uniform box with radius .25 gave ln 2 at (.5, .5, 0, 0) (TV .5).

        The minimum moves mass .25 into one class from one other class: H(.5, .25, .25, 0) = 1.5 ln 2.
        """
        nominal = np.full(4, 0.25, dtype=dtype)
        radius = np.array(0.25, dtype=dtype)

        value, p = lower_entropy(backend.distance(nominal, radius), return_distribution=True)

        np.testing.assert_allclose(backend.numpy(value), 1.5 * np.log(2), atol=tolerance(dtype))
        np.testing.assert_allclose(np.sort(backend.numpy(p)), [0.0, 0.25, 0.25, 0.5], atol=tolerance(dtype))
        assert_in_tv_ball(backend.numpy(p), nominal, radius, atol=tolerance(dtype))

    @DTYPES
    @pytest.mark.parametrize("n_classes", [2, 3, 4, 6, 10, 50])
    def test_upper_entropy_is_the_maximum(self, backend: Backend, dtype: type[np.floating], n_classes: int) -> None:
        """The maximizer is in the ball, no point of the ball is better, and SLSQP finds nothing better."""
        rng = np.random.default_rng(n_classes)
        balls = [random_tv_ball(rng, kind, n_classes) for kind in _TV_KINDS * 3]
        nominal = np.stack([b[0] for b in balls]).astype(dtype)
        radius = np.array([b[1] for b in balls], dtype=dtype)

        value, p = upper_entropy(backend.distance(nominal, radius), return_distribution=True)
        value, p = backend.numpy(value), backend.numpy(p)

        assert_in_tv_ball(p, nominal, radius, atol=tolerance(dtype))
        np.testing.assert_allclose(entropy(p), value, atol=tolerance(dtype))
        certificate = 1e-9 if dtype == np.float64 else 1e-4
        for row in range(len(balls)):
            assert tv_ball_ascent_gap(p[row], nominal[row], radius[row]) <= certificate, row
            if n_classes <= 10:
                reference = tv_ball_max_entropy(nominal[row], radius[row])
                assert value[row] >= reference - tolerance(dtype) - 1e-9, (row, value[row], reference)

    @DTYPES
    @pytest.mark.parametrize("n_classes", [2, 3, 4, 5])
    def test_lower_entropy_matches_vertex_enumeration(
        self, backend: Backend, dtype: type[np.floating], n_classes: int
    ) -> None:
        """The minimum over all vertices of the ball, enumerated from its inequalities."""
        rng = np.random.default_rng(10 + n_classes)
        balls = [random_tv_ball(rng, kind, n_classes) for kind in _TV_KINDS * 3]
        nominal = np.stack([b[0] for b in balls]).astype(dtype)
        radius = np.array([b[1] for b in balls], dtype=dtype)

        value, p = lower_entropy(backend.distance(nominal, radius), return_distribution=True)
        value, p = backend.numpy(value), backend.numpy(p)

        expected = [float(entropy(tv_ball_vertices(c, r)).min()) for c, r in zip(nominal, radius, strict=True)]
        np.testing.assert_allclose(value, expected, atol=tolerance(dtype))
        assert_in_tv_ball(p, nominal, radius, atol=tolerance(dtype))
        np.testing.assert_allclose(entropy(p), value, atol=tolerance(dtype))

    @DTYPES
    @pytest.mark.parametrize("n_classes", [6, 8])
    def test_lower_entropy_matches_all_receivers_and_donors(
        self, backend: Backend, dtype: type[np.floating], n_classes: int
    ) -> None:
        """The minimum over every choice of receiving, emptied and partial donor classes."""
        rng = np.random.default_rng(20 + n_classes)
        balls = [random_tv_ball(rng, kind, n_classes) for kind in _TV_KINDS * 2]
        nominal = np.stack([b[0] for b in balls]).astype(dtype)
        radius = np.array([b[1] for b in balls], dtype=dtype)

        value, p = lower_entropy(backend.distance(nominal, radius), return_distribution=True)
        value, p = backend.numpy(value), backend.numpy(p)

        expected = [tv_ball_min_entropy(c, r) for c, r in zip(nominal, radius, strict=True)]
        np.testing.assert_allclose(value, expected, atol=tolerance(dtype))
        assert_in_tv_ball(p, nominal, radius, atol=tolerance(dtype))

    @DTYPES
    @pytest.mark.parametrize("n_classes", [12, 50, 100])
    def test_lower_entropy_is_exact_for_many_classes(
        self, backend: Backend, dtype: type[np.floating], n_classes: int
    ) -> None:
        """The minimum over the boxes of every receiving class, which needs no threshold on the class count."""
        rng = np.random.default_rng(30 + n_classes)
        balls = [random_tv_ball(rng, kind, n_classes) for kind in _TV_KINDS * 3]
        nominal = np.stack([b[0] for b in balls]).astype(dtype)
        radius = np.array([b[1] for b in balls], dtype=dtype)

        value, p = lower_entropy(backend.distance(nominal, radius), return_distribution=True)
        value, p = backend.numpy(value), backend.numpy(p)

        expected = [tv_ball_min_entropy_by_receiver(c, r) for c, r in zip(nominal, radius, strict=True)]
        np.testing.assert_allclose(value, expected, atol=tolerance(dtype))
        assert_in_tv_ball(p, nominal, radius, atol=tolerance(dtype))
        np.testing.assert_allclose(entropy(p), value, atol=tolerance(dtype))

    @DTYPES
    def test_conformal_style_ball(self, backend: Backend, dtype: type[np.floating]) -> None:
        """Regression: ten classes around a softmax output with radius .2, as a TV conformal set produces.

        The box gave an upper entropy of 2.2048 at a point with TV .378; the maximum over
        the ball is 2.0847.
        """
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(5, 10)) * 2
        nominal = (np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)).astype(dtype)
        radius = np.full(5, 0.2, dtype=dtype)
        credal_set = backend.distance(nominal, radius)

        upper_value, upper_p = upper_entropy(credal_set, return_distribution=True)
        lower_value, lower_p = lower_entropy(credal_set, return_distribution=True)

        upper_value, upper_p = backend.numpy(upper_value), backend.numpy(upper_p)
        np.testing.assert_allclose(upper_value[0], 2.0847, atol=1e-4)
        certificate = 1e-9 if dtype == np.float64 else 1e-4
        for row in range(len(nominal)):
            assert upper_value[row] >= tv_ball_max_entropy(nominal[row], radius[row]) - tolerance(dtype) - 1e-9, row
            assert tv_ball_ascent_gap(upper_p[row], nominal[row], radius[row]) <= certificate, row
        expected_lower = [tv_ball_min_entropy(c, r) for c, r in zip(nominal[:2], radius[:2], strict=True)]
        np.testing.assert_allclose(backend.numpy(lower_value)[:2], expected_lower, atol=tolerance(dtype))
        expected_lower = [tv_ball_min_entropy_by_receiver(c, float(r)) for c, r in zip(nominal, radius, strict=True)]
        np.testing.assert_allclose(backend.numpy(lower_value), expected_lower, atol=tolerance(dtype))
        assert_in_tv_ball(upper_p, nominal, radius, atol=tolerance(dtype))
        assert_in_tv_ball(backend.numpy(lower_p), nominal, radius, atol=tolerance(dtype))

    @DTYPES
    def test_edge_cases(self, backend: Backend, dtype: type[np.floating]) -> None:
        """Radius zero, radii that reach the uniform distribution or a corner, and one-hot or sparse nominals."""
        nominal = np.array(
            [
                [0.5, 0.3, 0.2, 0.0],  # radius 0: the ball is the nominal point.
                [0.4, 0.3, 0.2, 0.1],  # radius .2 reaches the uniform distribution (TV .2).
                [0.4, 0.3, 0.2, 0.1],  # radius .6 reaches the corner (1, 0, 0, 0).
                [1.0, 0.0, 0.0, 0.0],  # One-hot: the minimum stays at the corner.
                [0.7, 0.3, 0.0, 0.0],  # Radius .1 raises both empty classes to .05.
            ],
            dtype=dtype,
        )
        radius = np.array([0.0, 0.2, 0.6, 0.3, 0.1], dtype=dtype)
        credal_set = backend.distance(nominal, radius)

        upper_value, upper_p = upper_entropy(credal_set, return_distribution=True)
        lower_value, lower_p = lower_entropy(credal_set, return_distribution=True)

        np.testing.assert_allclose(
            backend.numpy(upper_value),
            [
                entropy(nominal[0]),
                np.log(4),
                np.log(4),
                entropy(np.array([0.7, 0.1, 0.1, 0.1])),
                entropy(np.array([0.6, 0.3, 0.05, 0.05])),
            ],
            atol=tolerance(dtype),
        )
        np.testing.assert_allclose(
            backend.numpy(lower_value),
            [
                entropy(nominal[0]),
                entropy(np.array([0.6, 0.3, 0.1, 0.0])),
                0.0,
                0.0,
                entropy(np.array([0.8, 0.2, 0.0, 0.0])),
            ],
            atol=tolerance(dtype),
        )
        np.testing.assert_allclose(backend.numpy(upper_p)[0], nominal[0], atol=tolerance(dtype))
        np.testing.assert_allclose(backend.numpy(lower_p)[0], nominal[0], atol=tolerance(dtype))
        np.testing.assert_allclose(backend.numpy(lower_p)[2], [1.0, 0.0, 0.0, 0.0], atol=tolerance(dtype))
        assert_in_tv_ball(backend.numpy(upper_p), nominal, radius, atol=tolerance(dtype))
        assert_in_tv_ball(backend.numpy(lower_p), nominal, radius, atol=tolerance(dtype))

    @DTYPES
    def test_negative_radius_counts_as_zero(self, backend: Backend, dtype: type[np.floating]) -> None:
        """A negative radius gives the nominal distribution and its entropy, like radius zero."""
        nominal = np.array([[0.5, 0.3, 0.2, 0.0], [0.4, 0.3, 0.2, 0.1]], dtype=dtype)
        credal_set = backend.distance(nominal, np.array([-0.2, -1.0], dtype=dtype))

        for measure in (upper_entropy, lower_entropy):
            value, p = measure(credal_set, return_distribution=True)
            np.testing.assert_allclose(backend.numpy(value), entropy(nominal), atol=tolerance(dtype))
            np.testing.assert_allclose(backend.numpy(p), nominal, atol=tolerance(dtype))

    @DTYPES
    @pytest.mark.parametrize("n_classes", [1, 2])
    def test_one_and_two_classes(self, backend: Backend, dtype: type[np.floating], n_classes: int) -> None:
        """With one class the ball is a point; with two classes the ball is an interval."""
        nominal = np.array([1.0], dtype=dtype) if n_classes == 1 else np.array([0.7, 0.3], dtype=dtype)
        radius = np.array(0.1, dtype=dtype)
        credal_set = backend.distance(nominal, radius)

        upper_value = float(backend.numpy(upper_entropy(credal_set)))
        lower_value = float(backend.numpy(lower_entropy(credal_set)))

        expected_upper = 0.0 if n_classes == 1 else float(entropy(np.array([0.6, 0.4])))
        expected_lower = 0.0 if n_classes == 1 else float(entropy(np.array([0.8, 0.2])))
        np.testing.assert_allclose([upper_value, lower_value], [expected_upper, expected_lower], atol=tolerance(dtype))

    def test_batch_shapes_and_scalar_radius(self, backend: Backend) -> None:
        """Multi-dimensional batches with per-set radii, a scalar radius, and unbatched sets agree row by row."""
        rng = np.random.default_rng(5)
        balls = [random_tv_ball(rng, "softmax", 6) for _ in range(6)]
        nominal = np.stack([b[0] for b in balls])
        radius = np.array([b[1] for b in balls])

        for measure in (upper_entropy, lower_entropy):
            batched = backend.numpy(measure(backend.distance(nominal.reshape(2, 3, 6), radius.reshape(2, 3))))
            shared = backend.numpy(measure(backend.distance(nominal, np.array(radius[0]))))
            single = [float(backend.numpy(measure(backend.distance(c, np.array(r))))) for c, r in balls]
            assert batched.shape == (2, 3)
            np.testing.assert_allclose(batched.reshape(-1), single, atol=1e-12)
            np.testing.assert_allclose(shared[0], single[0], atol=1e-12)
            expected_shared = [float(backend.numpy(measure(backend.distance(c, np.array(radius[0]))))) for c in nominal]
            np.testing.assert_allclose(shared, expected_shared, atol=1e-12)

    @DTYPES
    def test_base(self, backend: Backend, dtype: type[np.floating]) -> None:
        """The log base rescales both entropies and leaves the optimizers unchanged."""
        credal_set = backend.distance(np.array([0.5, 0.2, 0.2, 0.1], dtype=dtype), np.array(0.15, dtype=dtype))

        for measure in (upper_entropy, lower_entropy):
            natural, p_natural = measure(credal_set, return_distribution=True)
            base_two, p_base_two = measure(credal_set, base=2.0, return_distribution=True)
            normalized = measure(credal_set, base="normalize")
            np.testing.assert_allclose(backend.numpy(base_two), backend.numpy(natural) / np.log(2), rtol=1e-6)
            np.testing.assert_allclose(backend.numpy(normalized), backend.numpy(natural) / np.log(4), rtol=1e-6)
            np.testing.assert_allclose(backend.numpy(p_natural), backend.numpy(p_base_two))
