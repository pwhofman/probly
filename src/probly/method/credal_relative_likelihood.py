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
credal_relative_likelihood.__doc__ = """Create a Credal Relative Likelihood predictor from a base predictor.

Based on :cite:`lohrCredalPrediction2025`.

Args:
    base: The base classifier to replicate into a class-bias ensemble.
    num_members: Number of ensemble members, one per class by convention.
    reset_params: Whether to reset the parameters of each member. Default is True.
    tobias_value: Magnitude of the class-specific bias initialization. Default is 3.0.

Returns:
    The credal relative likelihood predictor outputting a ProbabilityIntervalsCredalSet.
"""
representer.register(CredalRelativeLikelihoodPredictor, ProbabilityIntervalsRepresenter)


__all__ = ["CredalRelativeLikelihoodPredictor", "credal_relative_likelihood"]
