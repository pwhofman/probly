"""Backend-agnostic correctness suite for the upper entropy of convex credal sets.

Every expected value comes from an independent numpy reference implemented in
this module, never from the code under test:

* ``convex_max_entropy`` maximizes the entropy over the mixture weights with
  SLSQP from several starting points.
* ``convex_ascent_gap`` bounds how far a point of the hull is below the maximum,
  which follows from the concavity of the entropy.
* ``convex_hull_distance`` checks by a linear program that a point lies in the hull.

Backend test modules subclass the suite and provide a ``convex_backend`` fixture
that builds credal sets from numpy arrays and converts results back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pytest
from scipy.optimize import linprog, minimize

from probly.quantification.measure.credal_set import upper_entropy

if TYPE_CHECKING:
    from collections.abc import Callable

DTYPES = pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])


def tolerance(dtype: type[np.floating]) -> float:
    """Absolute tolerance for comparing entropies computed in ``dtype`` with float64 references."""
    return 1e-10 if dtype == np.float64 else 1e-5


class Backend(Protocol):
    """Builds backend credal sets from numpy arrays and converts backend arrays back."""

    def convex(self, vertices: np.ndarray) -> Any:  # noqa: ANN401
        """Build a convex credal set from vertices of shape ``(..., n_vertices, n_classes)``."""
        ...

    def numpy(self, value: Any) -> np.ndarray:  # noqa: ANN401
        """Convert a backend array to numpy."""
        ...


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats along the last axis, with ``0 log 0 = 0``."""
    p = np.asarray(p, dtype=np.float64)
    safe = np.where(p > 0, p, 1.0)
    return -(p * np.log(safe)).sum(-1)


def random_vertices(rng: np.random.Generator, kind: str, n_vertices: int, n_classes: int) -> np.ndarray:
    """Draw the vertices of a convex credal set of a given kind, shape ``(n_vertices, n_classes)``.

    Kinds:
        ``ensemble``: softmax outputs of ensemble members around shared logits.
        ``dirichlet``: Dirichlet(1) draws.
        ``sparse``: Dirichlet(0.3) draws with small entries set exactly to zero, and a last
        class that is zero in every vertex.
        ``repeated``: Dirichlet(0.5) draws with the first vertex used twice.
    """
    if kind == "ensemble":
        logits = rng.normal(size=n_classes) * 2 + rng.normal(size=(n_vertices, n_classes))
        vertices = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    elif kind == "dirichlet":
        vertices = rng.dirichlet(np.ones(n_classes), size=n_vertices)
    elif kind == "sparse":
        vertices = rng.dirichlet(np.full(n_classes, 0.3), size=n_vertices)
        vertices = np.where(vertices < 0.5 / n_classes, 0.0, vertices)
        if n_classes > 1:
            vertices[:, -1] = 0.0
        vertices[vertices.sum(-1) == 0, 0] = 1.0
        vertices /= vertices.sum(-1, keepdims=True)
    elif kind == "repeated":
        vertices = rng.dirichlet(np.full(n_classes, 0.5), size=n_vertices)
        vertices[-1] = vertices[0]
    else:
        msg = f"Unknown kind {kind}."
        raise ValueError(msg)
    return vertices


def convex_max_entropy(vertices: np.ndarray) -> float:
    """Maximum entropy over the convex hull of ``vertices`` by SLSQP on the mixture weights.

    Entropy is concave in the weights, so every local maximum is global; several starting
    points (the uniform weights and one near each vertex) guard against early stops.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    n = vertices.shape[0]
    starts = [np.full(n, 1.0 / n), *(0.9 * np.eye(n)[i] + 0.1 / n for i in range(n))]
    best = float(entropy(vertices).max())
    for start in starts:
        result = minimize(
            lambda w: -entropy(np.clip(w @ vertices, 0.0, None)),
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"ftol": 1e-15, "maxiter": 1000},
        )
        weights = np.clip(result.x, 0.0, None)
        best = max(best, float(entropy(weights / weights.sum() @ vertices)))
    return best


def convex_ascent_gap(p: np.ndarray, vertices: np.ndarray) -> float:
    """Upper bound on ``max_{q in hull} H(q) - H(p)`` for a point ``p`` of the hull.

    Entropy is concave, so ``H(q) <= H(p) + g . (q - p)`` with ``g = -log(p) - 1``, and the
    right side is largest at a vertex of the hull.
    """
    p = np.asarray(p, dtype=np.float64)
    gradient = -np.log(np.maximum(p, np.finfo(np.float64).tiny)) - 1.0
    return float((np.asarray(vertices, dtype=np.float64) @ gradient).max() - p @ gradient)


def convex_hull_distance(p: np.ndarray, vertices: np.ndarray) -> float:
    """L1 distance from ``p`` to the convex hull of ``vertices``, by a linear program."""
    p = np.asarray(p, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    n, k = vertices.shape
    # Variables: weights w (n), then slacks s (k) with -s <= w @ vertices - p <= s.
    result = linprog(
        np.concatenate([np.zeros(n), np.ones(k)]),
        A_ub=np.block([[vertices.T, -np.eye(k)], [-vertices.T, -np.eye(k)]]),
        b_ub=np.concatenate([p, -p]),
        A_eq=np.concatenate([np.ones(n), np.zeros(k)])[None],
        b_eq=[1.0],
        bounds=[(0.0, None)] * (n + k),
        method="highs",
    )
    assert result.status == 0, result.message
    return float(result.fun)


_CONVEX_KINDS = ["ensemble", "dirichlet", "sparse", "repeated"]


class ConvexUpperEntropySuite:
    """Upper entropy of convex hulls against SLSQP, a gap certificate and hull membership."""

    convex_backend: Callable[..., Backend]

    @DTYPES
    @pytest.mark.parametrize(("n_vertices", "n_classes"), [(2, 3), (4, 3), (5, 10), (10, 10), (20, 5), (8, 50)])
    def test_upper_entropy_is_the_maximum(
        self, convex_backend: Backend, dtype: type[np.floating], n_vertices: int, n_classes: int
    ) -> None:
        """The maximizer lies in the hull, no vertex improves the linear bound, and SLSQP finds nothing better."""
        rng = np.random.default_rng(100 * n_vertices + n_classes)
        vertices = np.stack([random_vertices(rng, kind, n_vertices, n_classes) for kind in _CONVEX_KINDS * 2])
        credal_set = convex_backend.convex(vertices.astype(dtype))
        # The references use the vertices as the backend stores them.
        seen = convex_backend.numpy(credal_set.tensor.probabilities).astype(np.float64)

        value, p = upper_entropy(credal_set, return_distribution=True)
        value, p = convex_backend.numpy(value), convex_backend.numpy(p)

        certificate = 1e-9 if dtype == np.float64 else 1e-4
        np.testing.assert_allclose(entropy(p), value, atol=tolerance(dtype))
        for row in range(len(vertices)):
            assert convex_hull_distance(p[row], seen[row]) <= 10 * tolerance(dtype), row
            assert convex_ascent_gap(p[row], seen[row]) <= certificate, row
            assert value[row] >= convex_max_entropy(seen[row]) - tolerance(dtype), row

    @DTYPES
    def test_upper_entropy_does_not_depend_on_the_rest_of_the_batch(
        self, convex_backend: Backend, dtype: type[np.floating]
    ) -> None:
        """A set gives the same value alone, in a large batch and in a multi-dimensional batch."""
        rng = np.random.default_rng(11)
        vertices = np.stack([random_vertices(rng, kind, 10, 10) for kind in _CONVEX_KINDS * 250]).astype(dtype)

        batched_value, batched_p = upper_entropy(convex_backend.convex(vertices), return_distribution=True)
        batched_value, batched_p = convex_backend.numpy(batched_value), convex_backend.numpy(batched_p)
        nested = convex_backend.numpy(upper_entropy(convex_backend.convex(vertices.reshape(10, 100, 10, 10))))

        atol = 1e-6 if dtype == np.float64 else 1e-5
        np.testing.assert_allclose(nested.reshape(-1), batched_value, atol=atol)
        for row in [0, 1, 2, 3, 500, 999]:
            alone_value, alone_p = upper_entropy(convex_backend.convex(vertices[row]), return_distribution=True)
            np.testing.assert_allclose(convex_backend.numpy(alone_value), batched_value[row], atol=atol)
            np.testing.assert_allclose(convex_backend.numpy(alone_p), batched_p[row], atol=10 * atol)

    @DTYPES
    def test_upper_entropy_edge_cases(self, convex_backend: Backend, dtype: type[np.floating]) -> None:
        """One class, repeated vertices, the corners of the simplex, and a maximum at a vertex."""
        cases = [
            (np.ones((3, 1)), 0.0),
            (np.tile([0.2, 0.3, 0.5], (4, 1)), float(entropy(np.array([0.2, 0.3, 0.5])))),
            (np.eye(4), np.log(4)),
            (np.array([[1.0, 0.0], [0.8, 0.2]]), float(entropy(np.array([0.8, 0.2])))),
        ]
        for vertices, expected in cases:
            value, p = upper_entropy(convex_backend.convex(vertices.astype(dtype)), return_distribution=True)
            np.testing.assert_allclose(convex_backend.numpy(value), expected, atol=tolerance(dtype))
            assert convex_hull_distance(convex_backend.numpy(p), vertices) <= 10 * tolerance(dtype)
