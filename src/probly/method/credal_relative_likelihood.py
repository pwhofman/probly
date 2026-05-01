"""Credal relative likelihood method compatibility layer."""

from __future__ import annotations

from typing import Protocol

from probly.representer import ProbabilityIntervalsRepresenter, representer
from probly.transformation.class_bias_ensemble import ClassBiasEnsemblePredictor, class_bias_ensemble


class CredalRelativeLikelihoodPredictor[**In, Out](ClassBiasEnsemblePredictor[In, Out], Protocol):
    """A predictor routed through the credal relative likelihood representer."""


credal_relative_likelihood = CredalRelativeLikelihoodPredictor.register_factory(
    class_bias_ensemble,
    autocast_builtins=True,
)
representer.register(CredalRelativeLikelihoodPredictor, ProbabilityIntervalsRepresenter)


__all__ = ["CredalRelativeLikelihoodPredictor", "credal_relative_likelihood"]
