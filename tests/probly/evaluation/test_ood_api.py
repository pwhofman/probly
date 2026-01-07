"""Tests for the unified OOD evaluation API.

These tests focus on API behavior:
- correct routing
- return types
- supported metric specifications
- error handling

Metric correctness is tested via the underlying task functions elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.ood import evaluate_ood, parse_dynamic_metric


# ---------------------------------------------------------------------
# evaluate_ood: return types / backward compatibility
# ---------------------------------------------------------------------
def test_evaluate_ood_default_returns_float_auroc() -> None:
    in_distribution = np.array([0.1, 0.2, 0.3])
    out_distribution = np.array([0.8, 0.9, 1.0])

    result = evaluate_ood(in_distribution, out_distribution)

    assert isinstance(result, float)


def test_evaluate_ood_metrics_auroc_returns_float() -> None:
    in_distribution = np.array([0.1, 0.2, 0.3])
    out_distribution = np.array([0.8, 0.9, 1.0])

    result = evaluate_ood(in_distribution, out_distribution, metrics="auroc")

    assert isinstance(result, float)


def test_evaluate_ood_single_metric_string_returns_dict() -> None:
    in_distribution = np.array([0.1, 0.2, 0.3])
    out_distribution = np.array([0.8, 0.9, 1.0])

    result = evaluate_ood(in_distribution, out_distribution, metrics="aupr")

    assert isinstance(result, dict)
    assert set(result.keys()) == {"aupr"}
    assert isinstance(result["aupr"], float)


def test_evaluate_ood_all_metrics_includes_static() -> None:
    in_distribution = np.array([0.1, 0.2, 0.3])
    out_distribution = np.array([0.8, 0.9, 1.0])

    result = evaluate_ood(in_distribution, out_distribution, metrics="all")

    assert isinstance(result, dict)
    assert "auroc" in result
    assert "aupr" in result
    assert "fpr@95" in result

    for k in ["auroc", "aupr", "fpr@95"]:
        assert isinstance(result[k], float)


def test_evaluate_ood_unknown_metric_raises() -> None:
    in_distribution = np.array([0.1, 0.2])
    out_distribution = np.array([0.8, 0.9])

    with pytest.raises(ValueError, match="Unknown metric"):
        evaluate_ood(in_distribution, out_distribution, metrics=["not_a_metric"])


# ---------------------------------------------------------------------
# parse_dynamic_metric: supported formats + errors
# ---------------------------------------------------------------------
def test_parse_dynamic_metric_valid_specs() -> None:
    assert parse_dynamic_metric("fpr@0.8") == ("fpr", 0.8)
    assert parse_dynamic_metric("fnr@95%") == ("fnr", 0.95)
    assert parse_dynamic_metric("tnr@0.99") == ("tnr", 0.99)


def test_parse_dynamic_metric_raises_without_at() -> None:
    # In the new API, dynamic metrics must be specified as "metric@threshold".
    with pytest.raises(ValueError):
        parse_dynamic_metric("fpr")


def test_parse_dynamic_metric_raises_unknown_base() -> None:
    with pytest.raises(ValueError, match="Invalid metric specification"):
        parse_dynamic_metric("unknown@95%")


def test_parse_dynamic_metric_raises_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        parse_dynamic_metric("fpr@0")

    with pytest.raises(ValueError):
        parse_dynamic_metric("fpr@1.1")

    with pytest.raises(ValueError):
        parse_dynamic_metric("fpr@xx")


# ---------------------------------------------------------------------
# evaluate_ood: dynamic metric dispatch
# ---------------------------------------------------------------------
def test_evaluate_ood_dynamic_metrics_return_dict() -> None:
    rng = np.random.default_rng(42)
    in_distribution = rng.random(50)
    out_distribution = rng.random(50)

    result = evaluate_ood(
        in_distribution,
        out_distribution,
        metrics=["fpr@0.95", "tnr@0.95", "fnr@95%"],
    )

    assert isinstance(result, dict)
    for k in ["fpr@0.95", "tnr@0.95", "fnr@95%"]:
        assert k in result
        assert isinstance(result[k], float)
        assert 0.0 <= result[k] <= 1.0
