"""Prior network method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.representation.distribution import DirichletDistribution
from probly.transformation.dirichlet_exp_activation import (
    DirichletExpActivationPredictor,
    dirichlet_exp_activation,
)


@runtime_checkable
class PriorNetworkPredictor[**In, Out: DirichletDistribution](DirichletExpActivationPredictor[In, Out], Protocol):
    """A predictor routed through the prior network method API."""


prior_network = PriorNetworkPredictor.register_factory(dirichlet_exp_activation)
prior_network.__doc__ = """Create a Prior Network predictor from a base predictor.

Based on :cite:`malininPredictiveUncertaintyEstimation2018`.

Args:
    base: The base logit classifier to be wrapped with a Dirichlet exp activation.

Returns:
    The prior network predictor outputting a Dirichlet distribution.
"""

__all__ = ["PriorNetworkPredictor", "prior_network"]
