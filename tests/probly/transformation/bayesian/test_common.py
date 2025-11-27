from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import bayesian


def test_bayesian(dummy_predictor: Predictor) -> None:
    """Test basic functionality of the Bayesian transformation."""
    
    bayesian_predictor = bayesian(
        dummy_predictor,
        posterior_std=0.1,
        prior_mean=0.0,
        prior_std=1.0,
    )

    # bayesian() soll einen Predictor zurückgeben
    assert isinstance(bayesian_predictor, Predictor)

    # Typ bleibt gleich
    assert isinstance(bayesian_predictor, type(dummy_predictor))



