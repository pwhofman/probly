"""Backend-agnostic tests for the selective prediction evaluation task."""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.selective_prediction import evaluate_selective_prediction, selective_prediction
from probly.metrics.selective_prediction import augrc, aurc, coverage_at_risk, risk_at_coverage


def test_selective_prediction_unregistered_type_raises() -> None:
    with pytest.raises(NotImplementedError, match="selective_prediction"):
        selective_prediction(object(), object())


def test_evaluate_default_metrics() -> None:
    criterion = np.array([0.1, 0.5, 0.5, 0.9])
    losses = np.array([0.0, 1.0, 0.0, 1.0])
    result = evaluate_selective_prediction(criterion, losses)
    assert result == {"aurc": aurc(criterion, losses), "augrc": augrc(criterion, losses)}
    assert evaluate_selective_prediction(criterion, losses, "all") == result
    assert evaluate_selective_prediction(criterion, losses, "ALL") == result


def test_evaluate_working_points() -> None:
    criterion = np.array([0.1, 0.5, 0.5, 0.9])
    losses = np.array([0.0, 1.0, 0.0, 1.0])
    result = evaluate_selective_prediction(criterion, losses, ["risk@0.5", "Coverage@0.4"])
    risk, coverage = risk_at_coverage(criterion, losses, 0.5)
    best_coverage, best_risk = coverage_at_risk(criterion, losses, 0.4)
    assert result == {
        "risk@0.5": risk,
        "risk@0.5:coverage": coverage,
        "Coverage@0.4": best_coverage,
        "Coverage@0.4:risk": best_risk,
    }
    assert all(isinstance(value, float) for value in result.values())


def test_evaluate_percentage_targets() -> None:
    criterion = np.array([0.1, 0.5, 0.5, 0.9])
    losses = np.array([0.0, 1.0, 0.0, 1.0])
    result = evaluate_selective_prediction(criterion, losses, ["risk@50%", "coverage@ 40 %"])
    assert result == {
        "risk@50%": risk_at_coverage(criterion, losses, 0.5)[0],
        "risk@50%:coverage": risk_at_coverage(criterion, losses, 0.5)[1],
        "coverage@ 40 %": coverage_at_risk(criterion, losses, 0.4)[0],
        "coverage@ 40 %:risk": coverage_at_risk(criterion, losses, 0.4)[1],
    }


def test_evaluate_accepts_lists() -> None:
    criterion = [0.1, 0.5, 0.5, 0.9]
    losses = [0.0, 1.0, 0.0, 1.0]
    expected = evaluate_selective_prediction(np.array(criterion), np.array(losses), "all")
    assert evaluate_selective_prediction(criterion, losses, "all") == expected
    assert evaluate_selective_prediction(tuple(criterion), tuple(losses), "all") == expected


@pytest.mark.parametrize("name", ["auroc", "risk", "risk@", "risk@high", "risk@%", "fpr@0.9"])
def test_evaluate_unknown_metric_raises(name: str) -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        evaluate_selective_prediction(np.zeros(3), np.zeros(3), [name])


@pytest.mark.parametrize(
    ("name", "match"),
    [("risk@1.5", "coverage must be in"), ("risk@150%", "coverage must be in"), ("coverage@nan", "risk must be")],
)
def test_evaluate_invalid_target_raises(name: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_selective_prediction(np.zeros(3), np.zeros(3), name)


def test_evaluate_working_points_report_realized_coverage_and_risk() -> None:
    # Coverage 0.5 is not reachable because of the tie, so the risk is reported at 0.75.
    result = evaluate_selective_prediction([0.1, 0.5, 0.5, 0.9], [0.0, 1.0, 0.0, 1.0], ["risk@0.5"])
    assert result["risk@0.5:coverage"] == 0.75
    # No coverage meets the target, so the risk at it is NaN.
    result = evaluate_selective_prediction([0.1, 0.2], [1.0, 1.0], ["coverage@0.5"])
    assert result["coverage@0.5"] == 0.0
    assert np.isnan(result["coverage@0.5:risk"])
