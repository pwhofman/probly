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

from probly.evaluation.ood_api import evaluate_ood, parse_dynamic_metric, visualize_ood


def test_evaluate_ood_returns_float() -> None:
    in_distribution = np.array([0.9, 0.8, 0.95])
    out_distributuion = np.array([0.1, 0.2, 0.05])

    result = evaluate_ood(in_distribution, out_distributuion)

    assert isinstance(result, float)


def test_evaluate_ood_single_metric_string_returns_float() -> None:
    in_distribution = np.array([0.9, 0.8, 0.95])
    out_distribution = np.array([0.1, 0.2, 0.05])

    result = evaluate_ood(
        in_distribution,
        out_distribution,
        metrics="auroc",
    )

    assert isinstance(result, float)


def test_evaluate_ood_multiple_metrics_returns_dict() -> None:
    in_distribution = np.array([0.9, 0.8, 0.95])
    out_distribution = np.array([0.1, 0.2, 0.05])

    result = evaluate_ood(
        in_distribution,
        out_distribution,
        metrics=["auroc", "aupr"],
    )

    assert isinstance(result, dict)
    assert "auroc" in result
    assert "aupr" in result


def test_evaluate_ood_all_metrics_returns_dict() -> None:
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
        "probly.evaluation.ood_api.evaluate_ood",
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
        "probly.evaluation.ood_api.evaluate_ood",
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
