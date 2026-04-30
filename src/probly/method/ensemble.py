"""Ensemble method compatibility exports."""

from __future__ import annotations

from probly.transformation.ensemble import (
    EnsembleCategoricalDistributionPredictor,
    EnsembleDirichletDistributionPredictor,
    EnsemblePredictor,
    ensemble,
    ensemble_generator,
    register_ensemble_members,
)

__all__ = [
    "EnsembleCategoricalDistributionPredictor",
    "EnsembleDirichletDistributionPredictor",
    "EnsemblePredictor",
    "ensemble",
    "ensemble_generator",
    "register_ensemble_members",
]
