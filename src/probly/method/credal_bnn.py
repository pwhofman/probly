"""CredalBNN method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.transformation.bayesian_ensemble import BayesianEnsemblePredictor, bayesian_ensemble


@runtime_checkable
class CredalBNNPredictor[**In, Out](BayesianEnsemblePredictor[In, Out], Protocol):
    """A predictor routed through the CredalBNN representer."""


credal_bnn = CredalBNNPredictor.register_factory(bayesian_ensemble, autocast_builtins=True)

__all__ = ["CredalBNNPredictor", "credal_bnn"]
