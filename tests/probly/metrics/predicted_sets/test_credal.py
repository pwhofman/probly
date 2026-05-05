"""Credal-set evaluation tests for ``coverage`` and ``efficiency``.

Each registered credal-set subtype has its own coverage semantics; the tests
verify those rules on small hand-computed inputs and end-to-end through a
conformal predictor for the conformal-set path.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation import average_interval_width, coverage, efficiency
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)

from ._credal_suite import CredalSuite


def _categorical(probs: np.ndarray) -> ArrayCategoricalDistribution:
    return ArrayProbabilityCategoricalDistribution(probs)


@pytest.fixture
def array_fn():
    return np.asarray


@pytest.fixture
def make_convex():
    return lambda probs: ArrayConvexCredalSet(array=_categorical(np.asarray(probs)))


@pytest.fixture
def make_distance():
    return lambda nominal, radius: ArrayDistanceBasedCredalSet(
        nominal=_categorical(np.asarray(nominal)),
        radius=np.asarray(radius),
    )


@pytest.fixture
def make_intervals():
    return lambda lower, upper: ArrayProbabilityIntervalsCredalSet(
        lower_bounds=np.asarray(lower),
        upper_bounds=np.asarray(upper),
    )


class TestNumpy(CredalSuite):
    """NumPy implementation of the shared credal suite."""


class TestSingletonCredalSet:
    def test_coverage_collapses_to_top1(self) -> None:
        probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
        cs = ArraySingletonCredalSet(array=_categorical(probs))
        # argmax is [0, 1, 0]; truth [0, 1, 2] -> 2/3 correct.
        assert coverage(cs, np.array([0, 1, 2])) == pytest.approx(2 / 3)

    def test_efficiency_is_one(self) -> None:
        cs = ArraySingletonCredalSet(array=_categorical(np.array([[0.5, 0.5]])))
        assert efficiency(cs) == 1.0


class TestDiscreteCredalSet:
    def test_coverage_uses_any_vertex_argmax(self) -> None:
        # Sample 0: vertex argmaxes are [0, 0, 0]; truth 0 -> covered.
        # Sample 1: vertex argmaxes are [0, 0, 3]; truth 3 -> covered (last vertex).
        probs = np.array(
            [
                [[0.7, 0.1, 0.1, 0.1], [0.5, 0.3, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]],
                [[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]],
            ]
        )
        cs = ArrayDiscreteCredalSet(array=_categorical(probs))
        assert coverage(cs, np.array([0, 3])) == pytest.approx(1.0)

    def test_coverage_misses_when_no_vertex_picks_truth(self) -> None:
        probs = np.array(
            [
                [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]],
                [[0.1, 0.8, 0.1], [0.2, 0.7, 0.1]],
            ]
        )
        cs = ArrayDiscreteCredalSet(array=_categorical(probs))
        # Sample 0 truth is 1 but every vertex picks 0; sample 1 truth is 1 and is picked.
        assert coverage(cs, np.array([1, 1])) == pytest.approx(0.5)

    def test_efficiency_counts_distinct_argmax(self) -> None:
        # Sample 0: distinct argmaxes = {0}; sample 1: {0, 3}; mean = 1.5.
        probs = np.array(
            [
                [[0.7, 0.1, 0.1, 0.1], [0.5, 0.3, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]],
                [[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]],
            ]
        )
        cs = ArrayDiscreteCredalSet(array=_categorical(probs))
        assert efficiency(cs) == pytest.approx(1.5)


class TestConvexCredalSet:
    def test_uses_interval_dominance(self) -> None:
        probs = np.array(
            [
                [[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]],
            ]
        )
        cs = ArrayConvexCredalSet(array=_categorical(probs))
        # lower = [0.4, 0.3, 0.1]; upper = [0.6, 0.5, 0.1]; max(lower) = 0.4.
        # mask = upper >= 0.4 = [True, True, False]. truth 1 is in the set.
        assert coverage(cs, np.array([1])) == pytest.approx(1.0)
        assert efficiency(cs) == pytest.approx(2.0)


class TestDistanceBasedCredalSet:
    def test_coverage_and_efficiency(self) -> None:
        nominal = _categorical(np.array([[0.5, 0.3, 0.2]]))
        radius = np.array([0.1])
        cs = ArrayDistanceBasedCredalSet(nominal=nominal, radius=radius)
        # lower = clip(nominal - 0.1) = [0.4, 0.2, 0.1]; max=0.4
        # upper = clip(nominal + 0.1) = [0.6, 0.4, 0.3]; mask = upper >= 0.4 = [T, T, F]
        assert coverage(cs, np.array([0])) == pytest.approx(1.0)
        assert coverage(cs, np.array([2])) == pytest.approx(0.0)
        assert efficiency(cs) == pytest.approx(2.0)


class TestProbabilityIntervalsCredalSet:
    def test_coverage_and_efficiency(self) -> None:
        lower = np.array([[0.1, 0.4, 0.05]])
        upper = np.array([[0.5, 0.6, 0.2]])
        cs = ArrayProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
        # max(lower) = 0.4; mask = upper >= 0.4 = [T, T, F]; truth 0 is covered, truth 2 is not.
        assert coverage(cs, np.array([0])) == pytest.approx(1.0)
        assert coverage(cs, np.array([2])) == pytest.approx(0.0)
        assert efficiency(cs) == pytest.approx(2.0)

    def test_average_interval_width(self) -> None:
        lower = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
        upper = np.array([[0.4, 0.4, 0.4], [1.0, 1.0, 1.0]])
        cs = ArrayProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
        # widths: [[0.3, 0.2, 0.1], [1.0, 1.0, 1.0]]; mean = (0.6 + 3.0) / 6 = 0.6.
        assert average_interval_width(cs) == pytest.approx(0.6)


def test_unregistered_type_raises() -> None:
    """Falling through to the base flexdispatch raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="coverage is not implemented"):
        coverage(object(), 0)
    with pytest.raises(NotImplementedError, match="efficiency is not implemented"):
        efficiency(object())
    with pytest.raises(NotImplementedError, match="average_interval_width is not implemented"):
        average_interval_width(object())


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
def test_empty_input_returns_nan() -> None:
    """Coverage/efficiency on an empty batch returns ``nan`` (numpy mean convention)."""
    cs = ArrayProbabilityIntervalsCredalSet(
        lower_bounds=np.zeros((0, 3)),
        upper_bounds=np.ones((0, 3)),
    )
    assert np.isnan(coverage(cs, np.zeros((0,), dtype=int)))
    assert np.isnan(efficiency(cs))


def test_end_to_end_with_classification_conformal_predictor() -> None:
    """End-to-end conformal-classification path: fit -> calibrate -> predict -> evaluate."""
    pytest.importorskip("sklearn")
    from sklearn.datasets import load_digits  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    from probly.calibrator import calibrate  # noqa: PLC0415
    from probly.method.conformal import conformal_lac  # noqa: PLC0415
    from probly.representer import representer  # noqa: PLC0415

    x, y = load_digits(return_X_y=True)
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4, random_state=0)
    x_cal, x_test, y_cal, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=0)

    base = LogisticRegression(max_iter=500).fit(x_train, y_train)
    calibrated = calibrate(conformal_lac(base), 0.1, y_cal, x_cal)
    output = representer(calibrated).predict(x_test)

    cov = coverage(output, y_test)
    eff = efficiency(output)
    assert 0.0 <= cov <= 1.0
    assert eff >= 0.0
