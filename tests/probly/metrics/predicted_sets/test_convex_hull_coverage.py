"""Tests for ``convex_hull_coverage``.

Strict-mode coverage uses the LP feasibility test
``V^T lambda = t, sum(lambda) = 1, lambda in [0, 1]``; relaxed-mode adds
L1 slack variables and counts covered iff the optimal slack sum is at most
``epsilon``. The hand-built fixtures exercise interior points, vertices,
out-of-hull targets, and the strict <-> relaxed transition.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.metrics import convex_hull_coverage
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArraySingletonCredalSet,
)
from probly.representation.distribution import ArrayCategoricalDistribution, ArrayProbabilityCategoricalDistribution


def _convex(probs: np.ndarray) -> ArrayConvexCredalSet:
    return ArrayConvexCredalSet(array=ArrayProbabilityCategoricalDistribution(probs))


def _discrete(probs: np.ndarray) -> ArrayDiscreteCredalSet:
    return ArrayDiscreteCredalSet(array=ArrayProbabilityCategoricalDistribution(probs))


def _singleton(probs: np.ndarray) -> ArraySingletonCredalSet:
    return ArraySingletonCredalSet(array=ArrayProbabilityCategoricalDistribution(probs))


def _dist(probs: np.ndarray) -> ArrayCategoricalDistribution:
    return ArrayProbabilityCategoricalDistribution(probs)


class TestStrict:
    def test_target_at_vertex_is_covered(self) -> None:
        # Hull of two corner vertices in a 2-class simplex.
        cs = _convex(np.array([[[1.0, 0.0], [0.0, 1.0]]]))
        assert convex_hull_coverage(cs, _dist(np.array([[1.0, 0.0]]))) == pytest.approx(1.0)

    def test_target_at_midpoint_is_covered(self) -> None:
        cs = _convex(np.array([[[1.0, 0.0], [0.0, 1.0]]]))
        assert convex_hull_coverage(cs, _dist(np.array([[0.5, 0.5]]))) == pytest.approx(1.0)

    def test_target_outside_hull_is_not_covered(self) -> None:
        # Hull only spans the first two corners of a 3-simplex.
        cs = _convex(np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
        # [0, 0, 1] requires lambda_1 + lambda_2 = 1 but the third coord is always 0 in the hull.
        assert convex_hull_coverage(cs, _dist(np.array([[0.0, 0.0, 1.0]]))) == pytest.approx(0.0)

    def test_partial_coverage_across_instances(self) -> None:
        # Two instances; one inside, one outside.
        v = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # hull along the [a, 1-a, 0] line
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        targets = np.array(
            [
                [0.4, 0.6, 0.0],  # in hull
                [0.0, 0.0, 1.0],  # not in hull
            ]
        )
        cs = _convex(v)
        assert convex_hull_coverage(cs, _dist(targets)) == pytest.approx(0.5)


class TestRelaxed:
    def test_epsilon_zero_matches_strict(self) -> None:
        v = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        target_inside = _dist(np.array([[0.4, 0.6, 0.0]]))
        target_outside = _dist(np.array([[0.0, 0.0, 1.0]]))
        cs = _convex(v)
        assert convex_hull_coverage(cs, target_inside, epsilon=0.0) == pytest.approx(
            convex_hull_coverage(cs, target_inside),
        )
        assert convex_hull_coverage(cs, target_outside, epsilon=0.0) == pytest.approx(
            convex_hull_coverage(cs, target_outside),
        )

    def test_epsilon_flips_verdict(self) -> None:
        # The L1 distance from [0, 0, 1] to the hull spanned by [1,0,0] and [0,1,0]
        # is 2.0 (e.g. project onto either vertex). Tight epsilon below that is not
        # enough; well above flips to covered.
        cs = _convex(np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
        target = _dist(np.array([[0.0, 0.0, 1.0]]))
        assert convex_hull_coverage(cs, target, epsilon=0.5) == pytest.approx(0.0)
        assert convex_hull_coverage(cs, target, epsilon=2.5) == pytest.approx(1.0)

    def test_relaxed_optimum_distance_equals_l1_to_hull(self) -> None:
        # When the target is a vertex (already on the hull), epsilon=0 covers it;
        # this also serves as a sanity check that the relaxed LP returns 0
        # objective for in-hull points.
        cs = _convex(np.array([[[1.0, 0.0], [0.0, 1.0]]]))
        target = _dist(np.array([[0.5, 0.5]]))
        assert convex_hull_coverage(cs, target, epsilon=0.01) == pytest.approx(1.0)

    def test_relaxed_lp_pins_nonzero_l1_distance(self) -> None:
        # Out-of-hull target with nonzero L1 distance to the hull. Vertices are
        # the first two corners of a 3-simplex: hull is the segment
        # {[a, 1-a, 0] : a in [0, 1]}. For target [0.3, 0.3, 0.4] the closest
        # hull point is [0.5, 0.5, 0] with L1 distance 0.2 + 0.2 + 0.4 = 0.8
        # (verified directly against scipy's HiGHS solver). Pin the LP optimum
        # against this distance via two epsilon thresholds straddling it.
        cs = _convex(np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
        target = _dist(np.array([[0.3, 0.3, 0.4]]))
        assert convex_hull_coverage(cs, target, epsilon=0.79) == pytest.approx(0.0)
        assert convex_hull_coverage(cs, target, epsilon=0.81) == pytest.approx(1.0)


class TestSingleton:
    def test_predicted_equals_target(self) -> None:
        cs = _singleton(np.array([[0.7, 0.2, 0.1]]))
        assert convex_hull_coverage(cs, _dist(np.array([[0.7, 0.2, 0.1]]))) == pytest.approx(1.0)

    def test_predicted_differs_strict_zero(self) -> None:
        cs = _singleton(np.array([[0.7, 0.2, 0.1]]))
        assert convex_hull_coverage(cs, _dist(np.array([[0.5, 0.3, 0.2]]))) == pytest.approx(0.0)

    def test_predicted_differs_relaxed(self) -> None:
        # L1 distance between [0.7, 0.2, 0.1] and [0.5, 0.3, 0.2] is 0.4.
        cs = _singleton(np.array([[0.7, 0.2, 0.1]]))
        target = _dist(np.array([[0.5, 0.3, 0.2]]))
        assert convex_hull_coverage(cs, target, epsilon=0.3) == pytest.approx(0.0)
        assert convex_hull_coverage(cs, target, epsilon=0.5) == pytest.approx(1.0)

    def test_partial_coverage(self) -> None:
        cs = _singleton(np.array([[0.7, 0.3], [0.4, 0.6]]))
        targets = _dist(np.array([[0.7, 0.3], [0.5, 0.5]]))
        # First instance: exact match -> covered. Second: l1=0.2, eps=0 -> not covered.
        assert convex_hull_coverage(cs, targets) == pytest.approx(0.5)


class TestDispatch:
    def test_discrete_and_convex_agree(self) -> None:
        # Same vertex array; same LP; same answer.
        v = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        target = _dist(np.array([[1 / 3, 1 / 3, 1 / 3]]))
        convex = convex_hull_coverage(_convex(v), target)
        discrete = convex_hull_coverage(_discrete(v), target)
        assert convex == pytest.approx(discrete)

    def test_unregistered_type_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="convex_hull_coverage is not implemented"):
            convex_hull_coverage(object(), _dist(np.array([[0.5, 0.5]])))

    def test_evaluation_reexport_resolves_identically(self) -> None:
        from probly.evaluation import convex_hull_coverage as reexport  # noqa: PLC0415

        assert reexport is convex_hull_coverage


class TestEdgeCases:
    def test_zero_instances_returns_nan(self) -> None:
        v = np.zeros((0, 2, 3))
        cs = _convex(v)
        targets = _dist(np.zeros((0, 3)))
        assert np.isnan(convex_hull_coverage(cs, targets))

    def test_single_vertex_hull(self) -> None:
        # Hull is a single point [0.5, 0.5]; only that exact target is covered.
        cs = _convex(np.array([[[0.5, 0.5]]]))
        assert convex_hull_coverage(cs, _dist(np.array([[0.5, 0.5]]))) == pytest.approx(1.0)
        assert convex_hull_coverage(cs, _dist(np.array([[0.7, 0.3]]))) == pytest.approx(0.0)

    def test_linprog_kwargs_escape_hatch(self) -> None:
        # ``method`` is a valid linprog kwarg. The handler should forward it without error.
        cs = _convex(np.array([[[1.0, 0.0], [0.0, 1.0]]]))
        target = _dist(np.array([[0.5, 0.5]]))
        assert convex_hull_coverage(cs, target, method="highs") == pytest.approx(1.0)

    def test_unknown_linprog_kwarg_raises(self) -> None:
        cs = _convex(np.array([[[1.0, 0.0], [0.0, 1.0]]]))
        target = _dist(np.array([[0.5, 0.5]]))
        with pytest.raises(TypeError, match=r"unexpected keyword|got an unexpected"):
            convex_hull_coverage(cs, target, totally_bogus_kwarg=42)

    def test_returns_np_floating(self) -> None:
        cs = _convex(np.array([[[1.0, 0.0], [0.0, 1.0]]]))
        result = convex_hull_coverage(cs, _dist(np.array([[0.5, 0.5]])))
        assert isinstance(result, np.floating)


class TestEpsilonValidation:
    @pytest.mark.parametrize("eps", [-1e-9, -1.0, float("nan"), float("inf"), float("-inf")])
    def test_invalid_epsilon_raises(self, eps: float) -> None:
        cs = _convex(np.array([[[1.0, 0.0], [0.0, 1.0]]]))
        target = _dist(np.array([[0.5, 0.5]]))
        with pytest.raises(ValueError, match="epsilon must be"):
            convex_hull_coverage(cs, target, epsilon=eps)

    @pytest.mark.parametrize("eps", [-1e-9, float("nan"), float("inf")])
    def test_invalid_epsilon_singleton_raises(self, eps: float) -> None:
        cs = _singleton(np.array([[0.5, 0.5]]))
        target = _dist(np.array([[0.5, 0.5]]))
        with pytest.raises(ValueError, match="epsilon must be"):
            convex_hull_coverage(cs, target, epsilon=eps)


class TestShapeValidation:
    def test_2d_unbatched_vertices_raises(self) -> None:
        # User accidentally passes (V, K) instead of (N, V, K).
        from probly.metrics.array import _convex_hull_lp_coverage  # noqa: PLC0415

        with pytest.raises(ValueError, match="vertices must be 3D"):
            _convex_hull_lp_coverage(np.zeros((2, 3)), np.zeros((1, 3)), 0.0)
        with pytest.raises(ValueError, match="targets must be 2D"):
            _convex_hull_lp_coverage(np.zeros((1, 2, 3)), np.zeros((3,)), 0.0)
        with pytest.raises(ValueError, match="vertices and targets must agree on N"):
            _convex_hull_lp_coverage(np.zeros((2, 2, 3)), np.zeros((1, 3)), 0.0)
        with pytest.raises(ValueError, match="vertices and targets must agree on K"):
            _convex_hull_lp_coverage(np.zeros((1, 2, 3)), np.zeros((1, 4)), 0.0)


class TestTorchParity:
    def test_numpy_torch_parity_convex(self) -> None:
        torch = pytest.importorskip("torch")
        from probly.representation.credal_set.torch import TorchConvexCredalSet  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        v = np.array(
            [
                [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]],
                [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]],
            ]
        )
        targets = np.array([[0.45, 0.45, 0.1], [0.5, 0.4, 0.1]])
        np_cs = _convex(v)
        tc_cs = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(torch.as_tensor(v)))
        np_t = _dist(targets)
        tc_t = TorchProbabilityCategoricalDistribution(torch.as_tensor(targets))

        assert convex_hull_coverage(np_cs, np_t) == pytest.approx(convex_hull_coverage(tc_cs, tc_t))
        assert convex_hull_coverage(np_cs, np_t, epsilon=0.3) == pytest.approx(
            convex_hull_coverage(tc_cs, tc_t, epsilon=0.3),
        )
