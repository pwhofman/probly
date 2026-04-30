"""Prior network method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.representation.distribution import DirichletDistribution
from probly.transformation.dirichlet_exp_activation import (
    DirichletExpActivationPredictor,
    dirichlet_exp_activation,
    register,
)


@runtime_checkable
class PriorNetworkPredictor[**In, Out: DirichletDistribution](DirichletExpActivationPredictor[In, Out], Protocol):
    """A predictor routed through the prior network method API."""


prior_network = PriorNetworkPredictor.register_factory(dirichlet_exp_activation)

__all__ = ["PriorNetworkPredictor", "prior_network", "register"]
