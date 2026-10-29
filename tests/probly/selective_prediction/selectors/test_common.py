"""Backend-agnostic tests for selective predictors, run with NumPy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.stats import binom

from probly.calibrator import Calibrator, calibrate
from probly.selective_prediction import Selector, selector as make_selector
from probly.selective_prediction.selectors import selector_registry
from probly.selective_prediction.selectors._common import CoverageSelector, _binomial_upper_bound

if TYPE_CHECKING:
    pass


def test_nan_threshold_raises() -> None:
    with pytest.raises(ValueError, match="NaN"):
        make_selector(threshold=float("nan"))


def test_threshold_selector_works_on_plain_arrays() -> None:
    selector = make_selector(threshold=0.5)
    uncertainty = np.array([0.1, 0.5, 0.7, np.nan])

    np.testing.assert_array_equal(selector.select(uncertainty), [True, True, False, False])


@pytest.mark.parametrize("uncertainty", [np.float64(0.1), np.zeros((2, 2))])
def test_threshold_selector_rejects_non_one_dimensional_uncertainty(uncertainty: np.ndarray) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        make_selector(threshold=0.5).select(uncertainty)


def test_threshold_selector_rejects_non_array_uncertainty() -> None:
    with pytest.raises(TypeError, match="array"):
        make_selector(threshold=0.5).select([0.1, 0.9])


def test_selector_base_class_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Selector()


def _kth_smallest(values: np.ndarray, coverage: float) -> float:
    n = len(values)
    k = int(np.ceil((n + 1) * coverage))
    return float(np.sort(values)[k - 1])


@pytest.mark.parametrize(("n", "coverage"), [(19, 0.9), (50, 0.8), (99, 0.5), (200, 0.95), (10, 0.75)])
def test_coverage_threshold_is_order_statistic(n: int, coverage: float) -> None:
    kappa = np.random.default_rng(n).permutation(n).astype(float) / n
    selector = make_selector(coverage=coverage).calibrate(kappa)
    assert selector.threshold == _kth_smallest(kappa, coverage)
    assert isinstance(selector.threshold, float)
    accepted = selector.select(kappa)
    assert accepted.sum() == int(np.ceil((n + 1) * coverage))


def test_coverage_rank_matches_exact_arithmetic_on_a_grid() -> None:
    for n in range(1, 201):
        kappa = np.arange(n, dtype=float)
        for hundredths in range(1, 101):
            rank = -((-(n + 1) * hundredths) // 100)  # exact ceil((n + 1) * c) for c = hundredths / 100
            selector = make_selector(coverage=hundredths / 100)
            if rank > n and hundredths < 100:
                needed = -(-hundredths // (100 - hundredths))  # exact ceil(c / (1 - c))
                with pytest.raises(ValueError, match=f"at least {needed} are needed"):
                    selector.calibrate(kappa)
                assert needed > n, (n, hundredths)
                continue
            threshold = selector.calibrate(kappa).threshold
            assert threshold == (np.inf if rank > n else float(rank - 1)), (n, hundredths)


@pytest.mark.parametrize("n", [1, 3, 10])
def test_full_coverage_threshold_is_inf(n: int) -> None:
    selector = make_selector(coverage=1.0).calibrate(np.arange(n, dtype=float))
    assert selector.threshold == np.inf
    assert selector.select(np.array([1e9, np.nan])).tolist() == [True, False]


@pytest.mark.parametrize(("n", "coverage", "needed"), [(3, 0.8, 4), (8, 0.9, 9), (18, 0.95, 19)])
def test_coverage_raises_if_calibration_set_is_too_small(n: int, coverage: float, needed: int) -> None:
    selector = make_selector(coverage=coverage)
    with pytest.raises(ValueError, match=f"too few for coverage={coverage}: at least {needed} are needed"):
        selector.calibrate(np.arange(n, dtype=float))
    assert not selector.is_calibrated
    assert selector.calibrate(np.arange(needed, dtype=float)).threshold == needed - 1


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.5, float("nan")])
def test_coverage_must_be_in_unit_interval(coverage: float) -> None:
    with pytest.raises(ValueError, match="coverage"):
        make_selector(coverage=coverage)


def test_coverage_selector_requires_calibration_and_returns_self() -> None:
    selector = make_selector(coverage=0.9)
    assert selector.threshold is None
    assert not selector.is_calibrated
    with pytest.raises(ValueError, match="not calibrated"):
        selector.select(np.array([0.1]))
    assert selector.calibrate(np.linspace(0.0, 1.0, 20)) is selector
    assert selector.is_calibrated


@pytest.mark.parametrize(
    "bad",
    [np.array([]), np.array([0.1, np.nan]), np.zeros((2, 2))],
)
def test_coverage_calibration_rejects_invalid_arrays(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        make_selector(coverage=0.9).calibrate(bad)


def test_coverage_calibration_rejects_non_array() -> None:
    with pytest.raises(TypeError, match="array"):
        make_selector(coverage=0.9).calibrate([0.1, 0.2])


def test_coverage_ties_at_threshold_are_accepted() -> None:
    kappa = np.array([0.1] * 10 + [0.5] * 10)
    selector = make_selector(coverage=0.6).calibrate(kappa)
    assert selector.threshold == 0.5
    assert selector.select(kappa).all()


def test_coverage_guarantee_by_simulation() -> None:
    rng = np.random.default_rng(0)
    n, coverage, repeats = 19, 0.9, 20000
    realized = np.empty(repeats)
    for i in range(repeats):
        cal, test = rng.random(n), rng.random()
        realized[i] = make_selector(coverage=coverage).calibrate(cal).select(np.array([test]))[0]
    # Without ties the expected coverage is exactly ceil((n + 1) * c) / (n + 1) = 18 / 20.
    se = np.sqrt(0.9 * 0.1 / repeats)
    assert coverage - 4 * se <= realized.mean() <= coverage + 1 / (n + 1) + 4 * se


def test_coverage_selector_is_a_calibrator() -> None:
    selector = make_selector(coverage=0.9)
    assert isinstance(selector, Calibrator)
    assert calibrate(selector, np.linspace(0.0, 1.0, 20)) is selector
    assert selector.threshold is not None


@pytest.mark.parametrize("errors", [0, 1, 4, 17])
def test_binomial_upper_bound_inverts_the_binomial_cdf(errors: int) -> None:
    n, delta = 40, 0.05
    bound = _binomial_upper_bound(errors, n, delta)
    assert binom.cdf(errors, n, bound) == pytest.approx(delta)


def test_binomial_upper_bound_edge_cases() -> None:
    assert _binomial_upper_bound(10, 10, 0.1) == 1.0
    assert _binomial_upper_bound(0, 10, 0.1) == pytest.approx(1.0 - 0.1 ** (1 / 10))


def _reference_sgr(kappa: np.ndarray, losses: np.ndarray, risk: float, delta: float) -> tuple[float, float, list[int]]:
    """Plain transcription of Algorithm 1 of :cite:`geifmanSelectiveClassification2017`.

    Returns the last certified iterate.
    """
    n = len(kappa)
    ordered = np.sort(kappa)
    steps = int(np.ceil(np.log2(n)))
    z_min, z_max = 1, n
    certified = (-np.inf, np.nan)
    tested = []
    for _ in range(steps):
        z = int(np.ceil((z_min + z_max) / 2))
        tau = ordered[n - z]
        mask = kappa <= tau
        tested.append(int(mask.sum()))
        bound = _binomial_upper_bound(losses[mask].sum(), int(mask.sum()), delta / steps)
        if bound < risk:
            z_max = z
            certified = (tau, bound)
        else:
            z_min = z
    return certified[0], certified[1], tested


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("ties", [False, True])
def test_sgr_matches_reference_transcription(seed: int, ties: bool) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 400))
    kappa = rng.random(n)
    if ties:
        kappa = np.round(kappa, 1)
    losses = (rng.random(n) < 0.02 + 0.2 * kappa**2).astype(float)
    expected_threshold, expected_bound, _ = _reference_sgr(kappa, losses, 0.15, 0.2)
    selector = make_selector(risk=0.15, delta=0.2).calibrate(kappa, losses)
    assert selector.threshold == expected_threshold
    np.testing.assert_allclose(selector.bound, expected_bound)


def test_sgr_hand_made_case() -> None:
    kappa = np.arange(8) / 10
    losses = np.zeros(8)
    # Tested accepted counts: 4, 6, 7 (all certified), so the threshold is the 7th smallest criterion.
    assert _reference_sgr(kappa, losses, 0.5, 0.5)[2] == [4, 6, 7]
    selector = make_selector(risk=0.5, delta=0.5).calibrate(kappa, losses)
    assert selector.threshold == pytest.approx(0.6)
    assert selector.bound == pytest.approx(1.0 - (0.5 / 3) ** (1 / 7))
    np.testing.assert_array_equal(selector.select(kappa), kappa <= 0.6)


def test_sgr_returns_last_certified_threshold_not_last_tested() -> None:
    kappa = np.arange(16) / 16
    losses = np.zeros(16)
    losses[12:] = 1.0
    selector = make_selector(risk=0.3, delta=0.5).calibrate(kappa, losses)
    assert selector.threshold is not None
    assert selector.bound is not None
    assert selector.bound < 0.3


@pytest.mark.parametrize(("n", "risk"), [(1, 0.5), (2, 0.9), (50, 0.01)])
def test_sgr_raises_without_certifiable_threshold(n: int, risk: float) -> None:
    kappa = np.linspace(0.0, 1.0, n)
    selector = make_selector(risk=risk, delta=0.1)
    with pytest.raises(ValueError, match="No threshold could be certified"):
        selector.calibrate(kappa, np.ones(n))
    assert not selector.is_calibrated
    assert selector.bound is None


def test_sgr_failed_calibration_keeps_the_previous_fit() -> None:
    kappa = np.linspace(0.0, 1.0, 50)
    selector = make_selector(risk=0.5, delta=0.1).calibrate(kappa, np.zeros(50))
    fitted = (selector.threshold, selector.bound)
    with pytest.raises(ValueError, match="No threshold could be certified"):
        selector.calibrate(kappa, np.ones(50))
    assert (selector.threshold, selector.bound) == fitted


@pytest.mark.parametrize(("risk", "delta"), [(0.0, 0.1), (1.0, 0.1), (-0.1, 0.1), (0.1, 0.0), (0.1, 1.0)])
def test_sgr_risk_and_delta_must_be_in_open_unit_interval(risk: float, delta: float) -> None:
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        make_selector(risk=risk, delta=delta)


def test_sgr_requires_losses() -> None:
    with pytest.raises(TypeError, match="losses"):
        make_selector(risk=0.1, delta=0.1).calibrate(np.linspace(0.0, 1.0, 20))


@pytest.mark.parametrize(
    ("kappa", "losses", "error", "match"),
    [
        (np.zeros(5), np.zeros(4), ValueError, "same length"),
        (np.zeros(0), np.zeros(0), ValueError, "at least one"),
        (np.array([0.1, np.nan]), np.zeros(2), ValueError, "NaN"),
        (np.zeros(2), np.array([0.0, np.nan]), ValueError, "NaN"),
        (np.zeros(2), np.array([0.0, 0.5]), ValueError, "binary"),
        (np.zeros(2), np.array([0.0, 2.0]), ValueError, "binary"),
        (np.zeros((2, 2)), np.zeros(2), ValueError, "^uncertainty must be one-dimensional"),
        (np.zeros(2), np.zeros((2, 2)), ValueError, "^losses must be one-dimensional"),
        ([0.1, 0.2], np.zeros(2), TypeError, "^uncertainty must be an array"),
        (np.zeros(2), [0.0, 0.0], TypeError, "^losses must be an array"),
    ],
)
def test_sgr_calibration_rejects_invalid_input(kappa: Any, losses: Any, error: type[Exception], match: str) -> None:  # noqa: ANN401
    with pytest.raises(error, match=match):
        make_selector(risk=0.1, delta=0.1).calibrate(kappa, losses)


def test_sgr_requires_calibration_before_select() -> None:
    selector = make_selector(risk=0.1, delta=0.1)
    assert selector.threshold is None
    assert selector.bound is None
    assert not selector.is_calibrated
    with pytest.raises(ValueError, match="not calibrated"):
        selector.select(np.zeros(3))


def test_sgr_selector_is_a_calibrator() -> None:
    selector = make_selector(risk=0.2, delta=0.1)
    assert isinstance(selector, Calibrator)
    rng = np.random.default_rng(0)
    kappa = rng.random(500)
    assert calibrate(selector, kappa, np.zeros(500)) is selector
    assert selector.is_calibrated
    assert selector.threshold is not None
    assert selector.threshold > 0.0


def test_sgr_risk_guarantee_by_simulation() -> None:
    rng = np.random.default_rng(0)
    n, risk, delta, repeats = 500, 0.05, 0.2, 300
    # The error probability is 0.3 * kappa^2, so the selective risk at threshold t is 0.1 * t^2, at most 0.05 up to
    # t = sqrt(0.5).
    limit = np.sqrt(0.5)
    violations = 0
    for _ in range(repeats):
        kappa = rng.random(n)
        losses = (rng.random(n) < 0.3 * kappa**2).astype(float)
        try:
            threshold = make_selector(risk=risk, delta=delta).calibrate(kappa, losses).threshold
        except ValueError:
            continue  # Nothing certified: no threshold is returned, so the guarantee cannot be violated.
        violations += threshold > limit
    assert violations / repeats <= delta


def test_selector_builds_the_selector_of_its_keywords() -> None:
    assert repr(make_selector(threshold=0.3)) == "selector(threshold=0.3)"
    assert repr(make_selector(coverage=0.8)) == "selector(coverage=0.8)"
    assert repr(make_selector(risk=0.1, delta=0.2)) == "selector(risk=0.1, delta=0.2)"
    assert make_selector(threshold=0.3, coverage=None).threshold == 0.3


@pytest.mark.parametrize(
    ("rules", "match"),
    [
        ({}, "exactly one rule"),
        ({"threshold": 0.5, "coverage": 0.5}, "exactly one rule"),
        ({"coverage": 0.5, "risk": 0.1, "delta": 0.1}, "exactly one rule"),
        ({"delta": 0.1}, "given together"),
        ({"risk": 0.1}, "given together"),
    ],
)
def test_selector_keywords_must_form_one_rule(rules: dict[str, float], match: str) -> None:
    with pytest.raises(TypeError, match=match):
        make_selector(**rules)


@pytest.mark.parametrize(
    "rules",
    [
        {"threshold": float("nan")},
        {"coverage": 0.0},
        {"coverage": 1.5},
        {"risk": 1.0, "delta": 0.1},
        {"risk": 0.1, "delta": 0.0},
    ],
)
def test_selector_keywords_out_of_range_raise(rules: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"NaN|coverage|\(0, 1\)"):
        make_selector(**rules)


def test_selector_registry_lists_the_builtin_rules() -> None:
    assert set(selector_registry) == {"threshold", "coverage", "risk"}
    assert selector_registry["coverage"] is CoverageSelector


def test_custom_rule_is_reachable_through_its_registered_keyword() -> None:
    @selector_registry.register("top")
    class TopSelector(Selector):
        def __init__(self, top: int) -> None:
            self.top = top

        def select(self, uncertainty):
            return uncertainty < np.sort(uncertainty)[self.top]

    try:
        built = make_selector(top=2)

        assert isinstance(built, TopSelector)
        np.testing.assert_array_equal(built.select(np.array([0.3, 0.1, 0.2])), [False, True, True])
    finally:
        del selector_registry["top"]


def test_option_of_another_rule_is_reported_by_name() -> None:
    with pytest.raises(TypeError, match="'bogus' is not an option of the 'coverage' rule"):
        make_selector(coverage=0.9, bogus=0.1)
