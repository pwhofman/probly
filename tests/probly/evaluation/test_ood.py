"""Tests for OOD unified API.

These tests focus exclusively on API behavior:
- correct routing
- return types
- supported metric specifications
- error handling

Metric correctness is tested elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.ood import evaluate_ood, parse_dynamic_metric


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

    for k in ("auroc", "aupr", "fpr"):
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
