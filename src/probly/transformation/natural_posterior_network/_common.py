"""Shared implementation of Natural Posterior Networks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.predictor import EvidentialPredictor, predict, predict_raw
from probly.representation.distribution import DirichletDistribution, create_dirichlet_distribution_from_alphas
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


type CertaintyBudget = Literal["constant", "exp-half", "exp", "normal"]
"""Named schemes for the certainty budget added to ``log p(z)`` before exponentiation.

See :cite:`charpentierNaturalPosteriorNetwork2022` and the official reference
implementation (`borchero/natural-posterior-network`). ``"normal"`` keeps the
evidence ``O(1)`` regardless of the latent dimension and is the default.
"""


@runtime_checkable
class NaturalPosteriorNetworkPredictor[**In, Out: DirichletDistribution](EvidentialPredictor, Protocol):
    """Protocol for natural posterior network predictors."""


def budget_log_scale(budget: CertaintyBudget, dim: int) -> float:
    """Return the additive log-scale used by an :class:`EvidenceScaler`-equivalent budget.

    Mirrors the four schemes from the official NatPN reference implementation
    (`borchero/natural-posterior-network`, ``natpn/nn/scaler.py``):

    - ``"constant"``: ``0`` (raw flow log-density).
    - ``"exp-half"``: ``0.5 * H``.
    - ``"exp"``: ``H``.
    - ``"normal"``: ``0.5 * log(4*pi) * H`` (default; cancels the
      ``-H/2 * log(2*pi)`` term of an isotropic Gaussian so a typical sample
      yields ``O(1)`` evidence regardless of latent dimension).

    Args:
        budget: One of the four named budget schemes.
        dim: Latent space dimension ``H``.

    Returns:
        Additive constant to apply to ``log p(z)`` before exponentiation.

    Raises:
        ValueError: If ``budget`` is not one of the recognised schemes.
    """
    if budget == "constant":
        return 0.0
    if budget == "exp-half":
        return 0.5 * dim
    if budget == "exp":
        return float(dim)
    if budget == "normal":
        return 0.5 * math.log(4 * math.pi) * dim
    msg = f"Unknown certainty budget {budget!r}"
    raise ValueError(msg)


@flexdispatch
def natural_posterior_network_generator[**In, Out: DirichletDistribution](
    encoder: Predictor[In, Out],
    latent_dim: int,
    num_classes: int,
    num_flows: int = 8,
    certainty_budget: CertaintyBudget = "normal",
    alpha_prior: float = 1.0,
) -> NaturalPosteriorNetworkPredictor[In, Out]:
    """Return a natural posterior network given an encoder model."""
    msg = f"No natural posterior network registered for type {type(encoder)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@NaturalPosteriorNetworkPredictor.register_factory
def natural_posterior_network[**In, Out: DirichletDistribution](
    encoder: Predictor[In, Out],
    latent_dim: int,
    num_classes: int,
    num_flows: int = 8,
    certainty_budget: CertaintyBudget = "normal",
    alpha_prior: float = 1.0,
) -> NaturalPosteriorNetworkPredictor[In, Out]:
    """Create a Natural Posterior Network predictor based on :cite:`charpentierNaturalPosteriorNetwork2022`."""
    return natural_posterior_network_generator(
        encoder, latent_dim, num_classes, num_flows, certainty_budget, alpha_prior
    )


@predict.register(NaturalPosteriorNetworkPredictor)
def _[**In](
    predictor: NaturalPosteriorNetworkPredictor[In, DirichletDistribution], *args: In.args, **kwargs: In.kwargs
) -> DirichletDistribution:
    """Predict with a natural posterior network predictor."""
    return create_dirichlet_distribution_from_alphas(predict_raw(predictor, *args, **kwargs))
