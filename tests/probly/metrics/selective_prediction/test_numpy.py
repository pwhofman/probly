"""NumPy backend tests for the selective prediction metrics."""

from __future__ import annotations

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


def test_aurc_and_augrc_exact_values() -> None:
    # Trapezoids between the distinct points (0, 0), (0.25, 0), (0.75, 1/3) and (1, 1/2).
    assert np.isclose(aurc(CRITERION, LOSSES), 0.5 * (1 / 3) / 2 + 0.25 * (1 / 3 + 1 / 2) / 2)
    # Generalized risk at the same points: 0, 0, 1/4 and 1/2.
    assert np.isclose(augrc(CRITERION, LOSSES), 0.5 * (1 / 4) / 2 + 0.25 * (1 / 4 + 1 / 2) / 2)


@pytest.mark.parametrize("decimals", [None, 1])
def test_augrc_matches_closed_form_for_zero_one_loss(decimals: int | None) -> None:
    # AUGRC = (1 - AUROC_f) * acc * (1 - acc) + (1 - acc)^2 / 2 (Traub et al., 2024, Eq. 8), with or without
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


def test_risk_at_coverage_takes_smallest_reachable_coverage() -> None:
    # Coverage 0.5 is not reachable because of the tie, so the risk is taken at the next reachable coverage, 0.75.
    assert np.isclose(risk_at_coverage(CRITERION, LOSSES, 0.5), 1 / 3)
    assert risk_at_coverage(CRITERION, LOSSES, 0.25) == 0.0
    assert risk_at_coverage(CRITERION, LOSSES, 1.0) == 0.5


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.1])
def test_risk_at_coverage_invalid_coverage_raises(coverage: float) -> None:
    with pytest.raises(ValueError, match="coverage must be in"):
        risk_at_coverage(CRITERION, LOSSES, coverage)


def test_coverage_at_risk_exact_values() -> None:
    assert coverage_at_risk(CRITERION, LOSSES, 0.0) == 0.25
    assert coverage_at_risk(CRITERION, LOSSES, 0.4) == 0.75
    assert coverage_at_risk(CRITERION, LOSSES, 0.5) == 1.0


def test_coverage_at_risk_unreachable_risk_is_zero() -> None:
    assert coverage_at_risk(np.array([0.1, 0.2]), np.array([1.0, 1.0]), 0.5) == 0.0


def test_coverage_at_risk_checks_every_threshold() -> None:
    # The risk is 1, 1/2, 2/3 and 1/2. It is not monotone, so a search that stops at the first risk above 1/2
    # would return a coverage of 0.5 instead of 1.0.
    criterion = np.array([0.1, 0.2, 0.3, 0.4])
    losses = np.array([1.0, 0.0, 1.0, 0.0])
    assert coverage_at_risk(criterion, losses, 0.5) == 1.0
    assert coverage_at_risk(criterion, losses, 0.6) == 1.0


@pytest.mark.parametrize("risk", [-0.1, float("nan")])
def test_coverage_at_risk_invalid_risk_raises(risk: float) -> None:
    with pytest.raises(ValueError, match="risk must be at least 0"):
        coverage_at_risk(CRITERION, LOSSES, risk)


def test_metrics_return_floats() -> None:
    assert isinstance(aurc(CRITERION, LOSSES), float)
    assert isinstance(augrc(CRITERION, LOSSES), float)
    assert isinstance(risk_at_coverage(CRITERION, LOSSES, 0.5), float)
    assert isinstance(coverage_at_risk(CRITERION, LOSSES, 0.5), float)
