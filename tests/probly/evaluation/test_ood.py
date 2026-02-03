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

from probly.evaluation.ood import evaluate_ood, parse_dynamic_metric, visualize_ood


def test_evaluate_ood_default_returns_dict_auroc() -> None:
    in_distribution = np.array([0.9, 0.8, 0.95])
    out_distribution = np.array([0.1, 0.2, 0.05])

    result = evaluate_ood(in_distribution, out_distribution)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"auroc"}
    assert isinstance(result["auroc"], float)


def test_evaluate_ood_single_metric_string_returns_dict() -> None:
    in_distribution = np.array([0.9, 0.8, 0.95])
    out_distribution = np.array([0.1, 0.2, 0.05])

    result = evaluate_ood(in_distribution, out_distribution, metrics="auroc")

    assert isinstance(result, dict)
    assert set(result.keys()) == {"auroc"}
    assert isinstance(result["auroc"], float)


def test_evaluate_ood_all_metrics_includes_static_and_dynamic() -> None:
    in_distribution = np.array([0.9, 0.8, 0.95])
    out_distribution = np.array([0.1, 0.2, 0.05])

    result = evaluate_ood(in_distribution, out_distribution, metrics="all")

    assert isinstance(result, dict)
    assert "auroc" in result
    assert "aupr" in result
    assert "fpr" in result
    assert "fnr" in result

    for k in ("auroc", "aupr", "fpr", "fnr"):
        assert isinstance(result[k], float)


def test_evaluate_ood_unknown_metric_raises() -> None:
    in_distribution = np.array([0.9, 0.8])
    out_distribution = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="Unknown metric"):
        evaluate_ood(in_distribution, out_distribution, metrics=["not_a_metric"])


def test_evaluate_ood_dynamic_metric_parsing_via_api_spec() -> None:
    """API should accept dynamic spec strings like 'fpr@0.8' and 'fnr@95%'."""
    in_distribution = np.array([0.9, 0.8, 0.95])
    out_distribution = np.array([0.1, 0.2, 0.05])

    result = evaluate_ood(in_distribution, out_distribution, metrics=["fpr@0.8", "fnr@95%"])

    assert isinstance(result, dict)
    assert "fpr@0.8" in result
    assert "fnr@95%" in result
    assert isinstance(result["fpr@0.8"], float)
    assert isinstance(result["fnr@95%"], float)


def test_parse_dynamic_metric_default_threshold() -> None:
    base, threshold = parse_dynamic_metric("fpr")
    assert base == "fpr"
    assert threshold == 0.95


def test_parse_dynamic_metric_parses_percent() -> None:
    base, threshold = parse_dynamic_metric("fnr@95%")
    assert base == "fnr"
    assert threshold == 0.95


def test_parse_dynamic_metric_raises_unknown() -> None:
    with pytest.raises(ValueError, match="Invalid metric specification"):
        parse_dynamic_metric("unknown@95%")


def test_visualize_ood_returns_figures() -> None:
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


def test_visualize_ood_subset_of_plots() -> None:
    rng = np.random.default_rng(123)
    in_distribution = rng.random(50)
    out_distribution = rng.random(50)

    figures = visualize_ood(in_distribution, out_distribution, plot_types=["roc"])

    assert isinstance(figures, dict)
    assert "roc" in figures
    assert "hist" not in figures
    assert "pr" not in figures

    assert isinstance(figures["roc"], Figure)


def test_visualize_ood_respects_invert_scores_flag() -> None:
    """Just API behavior: both settings should return figures without error."""
    rng = np.random.default_rng(7)
    in_distribution = rng.random(30)
    out_distribution = rng.random(30)

    figs_invert = visualize_ood(in_distribution, out_distribution, plot_types=["roc", "pr"], invert_scores=True)
    figs_noinvert = visualize_ood(in_distribution, out_distribution, plot_types=["roc", "pr"], invert_scores=False)

    assert set(figs_invert.keys()) == {"roc", "pr"}
    assert set(figs_noinvert.keys()) == {"roc", "pr"}

    assert isinstance(figs_invert["roc"], Figure)
    assert isinstance(figs_invert["pr"], Figure)
    assert isinstance(figs_noinvert["roc"], Figure)
    assert isinstance(figs_noinvert["pr"], Figure)
