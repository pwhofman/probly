"""Tests for OOD unified API.

These tests focus exclusively on API behavior:
- correct routing
- return types
- supported metric specifications
- error handling

Metric correctness is tested elsewhere.
"""

from __future__ import annotations

from matplotlib.figure import Figure
import numpy as np
import pytest

from probly.evaluation.ood import (
    evaluate_ood,
    out_of_distribution_detection_aupr,
    out_of_distribution_detection_auroc,
    out_of_distribution_detection_fnr_at_x_tpr,
    out_of_distribution_detection_fpr_at_x_tpr,
    parse_dynamic_metric,
    visualize_ood,
)


def test_out_of_distribution_detection_shape() -> None:
    rng = np.random.default_rng()
    auroc = out_of_distribution_detection_auroc(rng.random(10), rng.random(10))
    assert isinstance(auroc, float)


def test_out_of_distribution_detection_order() -> None:
    in_distribution = np.linspace(0, 1, 10)
    out_distribution = np.linspace(0, 1, 10) + 1
    auroc = out_of_distribution_detection_auroc(in_distribution, out_distribution)
    assert np.isclose(auroc, 0.995)


def test_out_of_distribution_detection_aupr_shape() -> None:
    """Test that AUPR OOD detection returns a float."""
    rng = np.random.default_rng()
    aupr = out_of_distribution_detection_aupr(rng.random(10), rng.random(10))
    assert isinstance(aupr, float)


def test_out_of_distribution_detection_aupr_order() -> None:
    """Test that AUPR OOD detection gives high score when OOD clearly differs from ID."""
    in_distribution = np.linspace(0, 1, 10)
    out_distribution = np.linspace(0, 1, 10) + 1  # clearly separated distributions
    aupr = out_of_distribution_detection_aupr(in_distribution, out_distribution)
    assert aupr > 0.99


def test_fpr_at_tpr_simple_case() -> None:
    in_scores = np.array([0.1, 0.2, 0.6, 0.7])
    out_scores = np.array([0.3, 0.4, 0.8, 0.9])

    fpr = out_of_distribution_detection_fpr_at_x_tpr(in_scores, out_scores, tpr_target=0.95)

    assert np.isclose(fpr, 0.5)


def test_fpr_at_tpr_invalid_tpr_target() -> None:
    in_scores = np.array([0.1, 0.2])
    out_scores = np.array([0.8, 0.9])

    msg = r"tpr_target must be in the interval \(0, 1]"

    with pytest.raises(ValueError, match=msg):
        out_of_distribution_detection_fpr_at_x_tpr(in_scores, out_scores, tpr_target=0.0)

    with pytest.raises(ValueError, match=msg):
        out_of_distribution_detection_fpr_at_x_tpr(in_scores, out_scores, tpr_target=1.1)


def test_fpr_at_tpr_perfect_separation() -> None:
    in_scores = np.array([0.1, 0.2, 0.3, 0.4])
    out_scores = np.array([0.8, 0.9, 1.0, 1.1])

    fpr = out_of_distribution_detection_fpr_at_x_tpr(in_scores, out_scores)

    assert np.isclose(fpr, 0.0)


def test_fnr_at_95_returns_float() -> None:
    """Tests if the funtion returns floats."""
    in_distribution = np.array([0.1, 0.2, 0.3])
    out_distribution = np.array([0.8, 0.9, 1.0])

    fnr = out_of_distribution_detection_fnr_at_x_tpr(in_distribution, out_distribution)

    assert isinstance(fnr, float)


def test_fnr_zero_when_perfect_separation() -> None:
    """If ID scores are clearly lower than OOD scores, FN should be 0."""
    in_distribution = np.array([0.1, 0.2, 0.3])
    out_distribution = np.array([1.0, 0.9, 0.8])

    fnr = out_of_distribution_detection_fnr_at_x_tpr(in_distribution, out_distribution)
    assert fnr == 0.0


def test_fnr_with_partial_overlap() -> None:
    """With overlapping distributions, the FNR should be between 0 and 1."""
    in_distribution = np.array([0.1, 0.4, 0.6])
    out_distribution = np.array([0.3, 0.5, 0.9])

    fnr = out_of_distribution_detection_fnr_at_x_tpr(in_distribution, out_distribution)
    assert 0.0 <= fnr <= 1.0


def test_single_element_arrays() -> None:
    """Edge case: one ID sample and one OOD sample."""
    in_distribution = np.array([0.2])
    out_distribution = np.array([0.9])

    fnr = out_of_distribution_detection_fnr_at_x_tpr(in_distribution, out_distribution)
    assert fnr == 0.0


def test_fpr_at_95_tpr_returns_float() -> None:
    """Tests if the function returns floats."""
    in_dist = np.zeros(20)
    out_dist = np.ones(20)
    """No random floats to reduce the possibility of the code crashing."""

    fpr = out_of_distribution_detection_fpr_at_x_tpr(in_dist, out_dist)

    assert isinstance(fpr, float)


def test_fpr_at_95_tpr_handles_missing_exact_point() -> None:
    in_distribution = np.linspace(0, 1, 10)
    out_distribution = np.linspace(0, 1, 10)
    fpr = out_of_distribution_detection_fpr_at_x_tpr(in_distribution, out_distribution)
    assert 0.0 <= fpr <= 1.0


def test_fpr_at_95_tpr_perfect_separation() -> None:
    """Test if FPR@95TPR OOD values are greater than ID values."""
    rng = np.random.default_rng(42)
    in_distribution = rng.uniform(0.0, 0.4, size=10)
    out_distribution = rng.uniform(0.6, 1.0, size=10)

    result = out_of_distribution_detection_fpr_at_x_tpr(in_distribution, out_distribution)

    assert np.isclose(result, 0.0)


def test_fpr_at_95_tpr_complete_overlap() -> None:
    """Tests if FPR@95TPR OOD- and ID-values are identical."""
    in_distribution = np.array([0.5, 0.5, 0.5, 0.5])
    out_distribution = np.array([0.5, 0.5, 0.5, 0.5])

    result = out_of_distribution_detection_fpr_at_x_tpr(in_distribution, out_distribution)

    assert np.isclose(result, 1.0)


def test_evaluate_ood_default_returns_dict_auroc() -> None:
    in_distribution = np.array([0.9, 0.8, 0.95])
    out_distribution = np.array([0.1, 0.2, 0.05])

    result = evaluate_ood(in_distribution, out_distribution)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"auroc"}


def test_evaluate_ood_single_metric_string_returns_dict() -> None:
    in_distribution = np.array([0.9, 0.8, 0.95])
    out_distribution = np.array([0.1, 0.2, 0.05])

    result = evaluate_ood(
        in_distribution,
        out_distribution,
        metrics="auroc",
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"auroc"}


def test_evaluate_ood_all_metrics_includes_static_and_dynamic() -> None:
    in_distribution = np.array([0.9, 0.8, 0.95])
    out_distribution = np.array([0.1, 0.2, 0.05])

    result = evaluate_ood(
        in_distribution,
        out_distribution,
        metrics="all",
    )

    assert isinstance(result, dict)
    assert "auroc" in result
    assert "aupr" in result
    assert any(k.startswith("fpr") for k in result)
    assert any(k.startswith("fnr") for k in result)


def test_evaluate_ood_unknown_metric_raises() -> None:
    in_distribution = np.array([0.9, 0.8])
    out_distribution = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="not_a_metric"):
        evaluate_ood(
            in_distribution,
            out_distribution,
            metrics=["not_a_metric"],
        )


def test_parse_dynamic_metric_default_threshold() -> None:
    base, threshold = parse_dynamic_metric("fpr")

    assert base == "fpr"
    assert threshold == 0.95


def test_parse_dynamic_metric_raises_unknown() -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_dynamic_metric("unknown@95%")


def test_visualize_ood_returns_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    """visualize_ood should return a dict of figures without crashing."""

    def fake_evaluate_ood(
        _in_distribution: np.ndarray,
        _out_distribution: np.ndarray,
        metrics: None | str | list[str] = None,
    ) -> dict[str, float]:
        _ = metrics

        return {
            "auroc": 0.5,
            "aupr": 0.5,
            "fpr@95tpr": 0.1,
        }

    monkeypatch.setattr(
        "probly.evaluation.ood.evaluate_ood",
        fake_evaluate_ood,
    )

    rng = np.random.default_rng(42)
    in_distribution = rng.random(50)
    out_distribution = rng.random(50)

    figures = visualize_ood(in_distribution, out_distribution)

    assert isinstance(figures, dict)
    assert "hist" in figures
    assert "roc" in figures
    assert "pr" in figures

    for fig in figures.values():
        assert isinstance(fig, Figure)


def test_visualize_ood_subset_of_plots(monkeypatch: pytest.MonkeyPatch) -> None:
    """visualize_ood should respect plot_types."""

    def fake_evaluate_ood(
        _in_distribution: np.ndarray,
        _out_distribution: np.ndarray,
        metrics: None | str | list[str] = None,
    ) -> dict[str, float]:
        _ = metrics

        return {
            "auroc": 0.5,
            "aupr": 0.5,
            "fpr@95tpr": 0.1,
        }

    monkeypatch.setattr(
        "probly.evaluation.ood.evaluate_ood",
        fake_evaluate_ood,
    )

    rng = np.random.default_rng(123)
    in_distribution = rng.random(50)
    out_distribution = rng.random(50)

    figures = visualize_ood(
        in_distribution,
        out_distribution,
        plot_types=["roc"],
    )

    assert isinstance(figures, dict)
    assert "roc" in figures
    assert "hist" not in figures
    assert "pr" not in figures
