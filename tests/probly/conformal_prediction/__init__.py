"""Tests for the Conformal Prediction module imports and structure."""

from __future__ import annotations

from abc import ABC

import probly.conformal_prediction as cp
from probly.conformal_prediction import ConformalPredictor
from probly.conformal_prediction.methods import SplitConformalPredictor, predict_probs
import probly.conformal_prediction.methods.common
import probly.conformal_prediction.methods.split
from probly.conformal_prediction.scores import Score
from probly.conformal_prediction.scores.aps.common import APSScore, aps_score_func
import probly.conformal_prediction.scores.common
from probly.conformal_prediction.scores.lac import LACScore, accretive_completion, lac_score_func
import probly.conformal_prediction.scores.lac.common  # noqa: F401
from probly.conformal_prediction.utils.metrics import average_set_size, empirical_coverage
from probly.conformal_prediction.utils.quantile import calculate_quantile, calculate_weighted_quantile


def test_import_conformal_prediction() -> None:
    """Test that the main module can be imported."""
    # check that ConformalPredictor is accessible
    assert ConformalPredictor is not None

    # check that ConformalPredictor is an abstract base class
    assert issubclass(ConformalPredictor, ABC)


def test_import_methods() -> None:
    """Test that methods submodule can be imported."""
    assert SplitConformalPredictor is not None
    assert predict_probs is not None


def test_import_scores() -> None:
    """Test that scores submodule can be imported."""
    # check all exports are present
    assert APSScore is not None
    assert LACScore is not None
    assert Score is not None
    assert accretive_completion is not None
    assert aps_score_func is not None
    assert lac_score_func is not None


def test_import_utils() -> None:
    """Test that utilities can be imported."""
    assert average_set_size is not None
    assert empirical_coverage is not None
    assert calculate_quantile is not None
    assert calculate_weighted_quantile is not None


def test_module_structure() -> None:
    """Test the overall module structure."""
    # check submodules are accessible
    assert hasattr(cp, "methods")
    assert hasattr(cp, "scores")
    assert hasattr(cp, "utils")

    # check __all__ exports
    if hasattr(cp, "__all__"):
        assert "ConformalPredictor" in cp.__all__


def test_type_checking_imports() -> None:
    """Test that TYPE_CHECKING imports don't break runtime."""
    # verify that the module imported successfully with TYPE_CHECKING imports
    assert cp.__name__ == "probly.conformal_prediction"
    assert hasattr(cp, "ConformalPredictor")
    assert hasattr(cp, "methods")
    assert hasattr(cp, "scores")
    assert hasattr(cp, "utils")
