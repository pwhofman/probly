"""Shared implementation of Posterior Networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from probly.method.method import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor
from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from probly.predictor import Predictor
from probly.predictor import EvidentialPredictor


@runtime_checkable
class PosteriorNetworkPredictor[**In, Out](EvidentialPredictor, Protocol):
    """Protocol for posterior network predictors."""


@lazydispatch
def posterior_network_generator[**In, Out](
    encoder: Predictor[In, Out], latent_dim: int, num_classes: int, class_counts: list | None = None, num_flows: int = 6
) -> PosteriorNetworkPredictor[In, Out]:
    """Return a posterior network given an encoder model."""
    msg = f"No posterior network registered for type {type(encoder)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=None)
@PosteriorNetworkPredictor.register_factory
def posterior_network[**In, Out](
    encoder: Predictor[In, Out],
    latent_dim: int,
    num_classes: int,
    class_counts: list | None = None,
    num_flows: int = 6,
) -> PosteriorNetworkPredictor[In, Out]:
    """Create a Posterior Network predictor from an encoder based on :cite:`charpentierPosteriorNetwork2020`."""
    return posterior_network_generator(encoder, latent_dim, num_classes, class_counts, num_flows)
