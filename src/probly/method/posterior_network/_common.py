"""Shared implementation of Posterior Networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.method.method import predictor_transformation
from probly.predictor import EvidentialPredictor, predict, predict_raw
from probly.representation.distribution import DirichletDistribution, create_dirichlet_distribution_from_alphas

if TYPE_CHECKING:
    from probly.predictor import Predictor


@runtime_checkable
class PosteriorNetworkPredictor[**In, Out: DirichletDistribution](EvidentialPredictor, Protocol):
    """Protocol for posterior network predictors."""


@flexdispatch
def posterior_network_generator[**In, Out: DirichletDistribution](
    encoder: Predictor[In, Out], latent_dim: int, num_classes: int, class_counts: list | None = None, num_flows: int = 6
) -> PosteriorNetworkPredictor[In, Out]:
    """Return a posterior network given an encoder model."""
    msg = f"No posterior network registered for type {type(encoder)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@PosteriorNetworkPredictor.register_factory
def posterior_network[**In, Out: DirichletDistribution](
    encoder: Predictor[In, Out],
    latent_dim: int,
    num_classes: int,
    class_counts: list | None = None,
    num_flows: int = 6,
) -> PosteriorNetworkPredictor[In, Out]:
    """Create a Posterior Network predictor from an encoder based on :cite:`charpentierPosteriorNetwork2020`."""
    return posterior_network_generator(encoder, latent_dim, num_classes, class_counts, num_flows)


@predict.register(PosteriorNetworkPredictor)
def _[**In](
    predictor: PosteriorNetworkPredictor[In, DirichletDistribution], *args: In.args, **kwargs: In.kwargs
) -> DirichletDistribution:
    """Predict with a posterior network predictor."""
    return create_dirichlet_distribution_from_alphas(predict_raw(predictor, *args, **kwargs))
