"""Credal relative likelihood method compatibility layer."""

from __future__ import annotations

from typing import Protocol

from flextype import flexdispatch

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


def relative_likelihood_thresholds(alpha: float, num_members: int) -> list[float]:
    """Per-member relative-likelihood targets: uniform interpolation over ``[alpha, 1)``.

    Member 0 is the maximum-likelihood reference and receives no target; the remaining members get the targets
    ``alpha + (1 - alpha) * j / (num_members - 1)`` for ``j = 0, ..., num_members - 2``.

    Args:
        alpha: Lowest relative-likelihood target, in (0, 1].
        num_members: Ensemble size including the reference member.

    Returns:
        ``num_members - 1`` targets, ``alpha`` first; empty for a single-member ensemble.

    Raises:
        ValueError: If alpha is outside (0, 1] or num_members < 1.
    """
    if not 0.0 < alpha <= 1.0:
        msg = f"alpha must be in (0, 1], got {alpha}."
        raise ValueError(msg)
    if num_members < 1:
        msg = f"num_members must be >= 1, got {num_members}."
        raise ValueError(msg)
    num_remaining = num_members - 1
    return [alpha + (1.0 - alpha) * j / num_remaining for j in range(num_remaining)]


@flexdispatch
def train_credal_relative_likelihood[**In, Out](
    predictor: CredalRelativeLikelihoodPredictor[In, Out],
    train_data: object,
    *,
    alpha: float = 0.95,
    **kwargs: object,
) -> CredalRelativeLikelihoodPredictor[In, Out]:
    """Train a credal relative likelihood ensemble based on :cite:`lohrCredalPrediction2025`.

    Member 0 is the maximum-likelihood reference; each remaining member trains only until its relative likelihood
    reaches its target from ``relative_likelihood_thresholds(alpha, num_members)``. Registered backends: torch,
    with ``train_data`` a ``DataLoader`` and the keyword arguments of
    ``probly.method.credal_relative_likelihood.torch``.

    Args:
        predictor: The ``credal_relative_likelihood`` ensemble; members are trained in place.
        train_data: Backend-specific training data.
        alpha: Lowest relative-likelihood target, in (0, 1]. Default is 0.95.
        **kwargs: Backend-specific keyword arguments.

    Returns:
        The trained predictor.

    Raises:
        NotImplementedError: If no trainer is registered for the predictor's backend.
    """
    msg = f"No credal relative likelihood trainer is registered for type {type(predictor)}"
    raise NotImplementedError(msg)


representer.register(CredalRelativeLikelihoodPredictor, ProbabilityIntervalsRepresenter)


__all__ = [
    "CredalRelativeLikelihoodPredictor",
    "credal_relative_likelihood",
    "relative_likelihood_thresholds",
    "train_credal_relative_likelihood",
]
