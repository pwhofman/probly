"""NumPy backend tests for the selective prediction metrics."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from probly.metrics import roc_auc_score
from probly.metrics.selective_prediction import (
    augrc,
    aurc,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Two tied criterion values in the middle: they are accepted together.
CRITERION = np.array([0.5, 0.1, 0.9, 0.5])
LOSSES = np.array([1.0, 0.0, 1.0, 0.0])


def test_risk_coverage_curve_exact_values() -> None:
    coverage, risk, thresholds = risk_coverage_curve(CRITERION, LOSSES)
    np.testing.assert_allclose(coverage, [0.0, 0.25, 0.75, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0.0, 0.0, 1 / 3, 1 / 3, 0.5])
    np.testing.assert_array_equal(thresholds, [-np.inf, 0.1, 0.5, 0.5, 0.9])


def test_risk_coverage_curve_endpoint_carries_first_risk() -> None:
    _, risk, _ = risk_coverage_curve(np.array([0.2, 0.2, 0.7]), np.array([1.0, 0.0, 0.0]))
    assert risk[0] == risk[1] == 0.5


def test_risk_coverage_curve_does_not_depend_on_order_under_ties() -> None:
    rng = np.random.default_rng(0)
    criterion = np.round(rng.random(200), 1)
    losses = rng.random(200)
    permutation = rng.permutation(200)
    for expected, actual in zip(
        risk_coverage_curve(criterion, losses),
        risk_coverage_curve(criterion[permutation], losses[permutation]),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("criterion", "losses"),
    [(np.zeros((2, 3)), np.zeros((2, 3))), (np.zeros(3), np.zeros(4)), (np.zeros(0), np.zeros(0))],
)
def test_risk_coverage_curve_invalid_shapes_raise(criterion: np.ndarray, losses: np.ndarray) -> None:
    with pytest.raises(ValueError, match="one-dimensional with the same, nonzero length"):
        risk_coverage_curve(criterion, losses)


def test_risk_coverage_curve_finds_ties_before_casting() -> None:
    # 2**53 and 2**53 + 1 are distinct integers but equal as float64, so they must form two steps, not one.
    coverage, _, _ = risk_coverage_curve(np.array([2**53, 2**53 + 1]), np.array([1.0, 0.0]))
    np.testing.assert_array_equal(coverage, [0.0, 0.5, 1.0])


def test_risk_coverage_curve_accepts_losses_as_list() -> None:
    expected = risk_coverage_curve(CRITERION, LOSSES)
    for e, a in zip(expected, risk_coverage_curve(CRITERION, LOSSES.tolist()), strict=True):
        np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize("function", [risk_coverage_curve, aurc, augrc])
def test_nan_criterion_raises(function: Callable[..., object]) -> None:
    with pytest.raises(ValueError, match="criterion must not contain NaN"):
        function(np.array([0.1, np.nan, 0.3]), np.zeros(3))


@pytest.mark.parametrize("function", [risk_coverage_curve, aurc, augrc])
def test_negative_infinite_criterion_raises(function: Callable[..., object]) -> None:
    with pytest.raises(ValueError, match="criterion must not contain -inf"):
        function(np.array([-np.inf, 0.2, 0.3]), np.zeros(3))


def test_positive_infinite_criterion_is_allowed() -> None:
    coverage, _, thresholds = risk_coverage_curve(np.array([0.2, np.inf]), np.array([0.0, 1.0]))
    np.testing.assert_allclose(coverage, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(thresholds, [-np.inf, 0.2, np.inf])


@pytest.mark.parametrize(
    "function",
    [
        risk_coverage_curve,
        aurc,
        augrc,
        lambda criterion, losses: risk_at_coverage(criterion, losses, 0.25),
        lambda criterion, losses: coverage_at_risk(criterion, losses, 0.5),
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_losses_raise(function: Callable[..., object], value: float) -> None:
    # Without the check, coverage_at_risk returned 0.5 for NaN, because every comparison with NaN is false, and
    # aurc returned NaN for an infinite loss, from inf - inf in the expectation over tie orders.
    with pytest.raises(ValueError, match="losses must be finite"):
        function(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 1.0, value, 0.0]))


def test_aurc_and_augrc_exact_values() -> None:
    # Inside the tied step, half of the tied instances are accepted at coverage 0.5, with an expected loss of 1/2,
    # so the expected risk there is 1/4. Trapezoids between (0, 0), (0.25, 0), (0.5, 1/4), (0.75, 1/3) and
    # (1, 1/2).
    expected = 0.25 * (1 / 4) / 2 + 0.25 * (1 / 4 + 1 / 3) / 2 + 0.25 * (1 / 3 + 1 / 2) / 2
    assert np.isclose(aurc(CRITERION, LOSSES), expected)
    # Generalized risk at the distinct points (0, 0), (0.25, 0), (0.75, 1/4) and (1, 1/2).
    assert np.isclose(augrc(CRITERION, LOSSES), 0.5 * (1 / 4) / 2 + 0.25 * (1 / 4 + 1 / 2) / 2)


def _per_order_aurc(criterion: np.ndarray, losses: np.ndarray) -> float:
    # AURC of one fixed order: every instance is its own step, with the endpoint (0, first risk).
    order = np.argsort(criterion, kind="stable")
    count = np.arange(1, len(losses) + 1)
    risk = np.cumsum(losses[order]) / count
    return float(np.trapezoid(np.concatenate([risk[:1], risk]), np.concatenate([[0.0], count / len(losses)])))


def test_aurc_is_mean_over_orders_of_tied_instances() -> None:
    criterion = np.array([0.2, 0.7, 0.2, 0.7, 0.7, 0.4, 0.2])
    losses = np.array([0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.2])
    values = [
        _per_order_aurc(criterion[list(permutation)], losses[list(permutation)])
        for permutation in itertools.permutations(range(len(losses)))
    ]
    assert np.isclose(aurc(criterion, losses), np.mean(values))


def test_aurc_without_ties_matches_per_order_value() -> None:
    rng = np.random.default_rng(3)
    criterion = rng.random(100)
    losses = rng.random(100)
    assert np.isclose(aurc(criterion, losses), _per_order_aurc(criterion, losses))


def _reference_selective_risk_stats(
    criterion: np.ndarray, losses: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transcription of the curve construction of FD-Shifts (selective_risk_stats and generalized_risk_stats).

    Instances are sorted by decreasing confidence (confidence is minus the criterion). The risk after k instances is
    the running mean of the losses, the generalized risk is the running sum divided by n, and only the last position
    of each group of tied confidences is kept, because tied instances are accepted together.
    """
    n = len(losses)
    order = np.argsort(criterion, kind="stable")
    confidence = -criterion[order]
    ordered_losses = losses[order]
    counts = np.arange(1, n + 1)
    risks = np.cumsum(ordered_losses) / counts
    generalized = np.cumsum(ordered_losses) / n
    last_of_group = np.append(confidence[1:] != confidence[:-1], True)
    coverages = counts[last_of_group] / n
    return coverages, risks[last_of_group], generalized[last_of_group]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("ties", [False, True])
def test_curve_and_augrc_match_fd_shifts_reference(seed: int, ties: bool) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 300))
    criterion = rng.random(n)
    if ties:
        criterion = np.round(criterion, 1)
    losses = (rng.random(n) < 0.05 + 0.4 * criterion).astype(float)
    coverages, risks, generalized = _reference_selective_risk_stats(criterion, losses)

    coverage, risk, _ = risk_coverage_curve(criterion, losses)
    # probly has one point per instance, with tied instances sharing a coverage, and adds the endpoint (0, first
    # risk). Keep the last point of every coverage value to compare with the reference.
    assert coverage[0] == 0.0
    assert risk[0] == risks[0]
    last = np.append(coverage[1:-1] != coverage[2:], True)
    np.testing.assert_allclose(coverage[1:][last], coverages)
    np.testing.assert_allclose(risk[1:][last], risks)

    expected_augrc = np.trapezoid(np.concatenate([[0.0], generalized]), np.concatenate([[0.0], coverages]))
    assert np.isclose(augrc(criterion, losses), expected_augrc)
    if not ties:
        # Without ties the step construction and the expectation over orders agree with the plain trapezoid.
        expected_aurc = np.trapezoid(np.concatenate([risks[:1], risks]), np.concatenate([[0.0], coverages]))
        assert np.isclose(aurc(criterion, losses), expected_aurc)


def test_aurc_does_not_prefer_a_coarsened_criterion() -> None:
    # The coarsened copy splits the instances into the most confident 5 % and a tie of all others, so it
    # discards ranking information. A straight line inside the tied step would give it the smaller area here.
    rng = np.random.default_rng(6)
    criterion = rng.random(5000)
    losses = (rng.random(5000) < 0.05 + 0.4 * criterion).astype(float)
    coarse = (criterion > np.quantile(criterion, 0.05)).astype(float)
    assert aurc(coarse, losses) > aurc(criterion, losses)


@pytest.mark.parametrize("decimals", [None, 1])
def test_augrc_matches_closed_form_for_zero_one_loss(decimals: int | None) -> None:
    # AUGRC = (1 - AUROC_f) * acc * (1 - acc) + (1 - acc)^2 / 2 (Traub et al., 2024, Eq. 7), with or without
    # ties in the criterion.
    rng = np.random.default_rng(1)
    criterion = rng.random(300)
    if decimals is not None:
        criterion = np.round(criterion, decimals)
    losses = (rng.random(300) < 0.2 + 0.5 * criterion).astype(float)
    accuracy = 1 - losses.mean()
    auroc = float(roc_auc_score(losses, criterion))
    expected = (1 - auroc) * accuracy * (1 - accuracy) + 0.5 * (1 - accuracy) ** 2
    assert np.isclose(augrc(criterion, losses), expected)


@pytest.mark.parametrize("decimals", [None, 1])
def test_augrc_matches_pair_form_for_general_losses(decimals: int | None) -> None:
    # AUGRC = mean(losses) / 2 + sum over pairs (i accepted before j) of (l_i - l_j) / (2 n^2). Tied pairs add 0.
    rng = np.random.default_rng(2)
    n = 200
    criterion = rng.random(n)
    if decimals is not None:
        criterion = np.round(criterion, decimals)
    losses = rng.random(n) * (rng.random(n) < 0.7)
    before = criterion[:, None] < criterion[None, :]
    pair_term = np.sum(before * (losses[:, None] - losses[None, :])) / (2 * n**2)
    assert np.isclose(augrc(criterion, losses), losses.mean() / 2 + pair_term)


def test_risk_at_coverage_takes_smallest_reachable_coverage() -> None:
    # Coverage 0.5 is not reachable because of the tie, so the risk is taken at the next reachable coverage, 0.75.
    risk, coverage = risk_at_coverage(CRITERION, LOSSES, 0.5)
    assert np.isclose(risk, 1 / 3)
    assert coverage == 0.75
    assert risk_at_coverage(CRITERION, LOSSES, 0.25) == (0.0, 0.25)
    assert risk_at_coverage(CRITERION, LOSSES, 1.0) == (0.5, 1.0)


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.1])
def test_risk_at_coverage_invalid_coverage_raises(coverage: float) -> None:
    with pytest.raises(ValueError, match="coverage must be in"):
        risk_at_coverage(CRITERION, LOSSES, coverage)


def test_coverage_at_risk_exact_values() -> None:
    assert coverage_at_risk(CRITERION, LOSSES, 0.0) == (0.25, 0.0)
    coverage, risk = coverage_at_risk(CRITERION, LOSSES, 0.4)
    assert coverage == 0.75
    assert np.isclose(risk, 1 / 3)
    assert coverage_at_risk(CRITERION, LOSSES, 0.5) == (1.0, 0.5)


def test_coverage_at_risk_unreachable_risk_is_zero_coverage_and_nan_risk() -> None:
    coverage, risk = coverage_at_risk(np.array([0.1, 0.2]), np.array([1.0, 1.0]), 0.5)
    assert coverage == 0.0
    assert np.isnan(risk)


def test_coverage_at_risk_checks_every_threshold() -> None:
    # The risk is 1, 1/2, 2/3 and 1/2. It is not monotone, so a search that stops at the first risk above 1/2
    # would return a coverage of 0.5 instead of 1.0.
    criterion = np.array([0.1, 0.2, 0.3, 0.4])
    losses = np.array([1.0, 0.0, 1.0, 0.0])
    assert coverage_at_risk(criterion, losses, 0.5) == (1.0, 0.5)
    assert coverage_at_risk(criterion, losses, 0.6) == (1.0, 0.5)


@pytest.mark.parametrize("risk", [-0.1, float("nan")])
def test_coverage_at_risk_invalid_risk_raises(risk: float) -> None:
    with pytest.raises(ValueError, match="risk must be at least 0"):
        coverage_at_risk(CRITERION, LOSSES, risk)


def test_metrics_return_floats() -> None:
    assert isinstance(aurc(CRITERION, LOSSES), float)
    assert isinstance(augrc(CRITERION, LOSSES), float)
    for function in (risk_at_coverage, coverage_at_risk):
        assert all(isinstance(value, float) for value in function(CRITERION, LOSSES, 0.5))
