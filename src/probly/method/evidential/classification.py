"""Evidential classification method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.representation.distribution import DirichletDistribution
from probly.transformation.dirichlet_clipped_exp_one_activation import (
    DirichletClippedExpOneActivationPredictor,
    dirichlet_clipped_exp_one_activation,
    register,
)


@runtime_checkable
class EvidentialClassificationPredictor[**In, Out: DirichletDistribution](
    DirichletClippedExpOneActivationPredictor[In, Out], Protocol
):
    """A predictor routed through the evidential classification method API."""


evidential_classification = EvidentialClassificationPredictor.register_factory(
    dirichlet_clipped_exp_one_activation,
)

__all__ = ["EvidentialClassificationPredictor", "evidential_classification", "register"]
