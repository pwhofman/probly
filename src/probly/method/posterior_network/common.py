"""Shared implementation of Posterior Networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probly.predictor import Predictor
from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from probly.predictor import Predictor


@lazydispatch
def posterior_network_generator[**In, Out](
    encoder: Predictor[In, Out], dim: int, num_classes: int, class_counts: list | None = None, num_flows: int = 6
) -> Predictor[In, Out]:
    """Return a posterior network given an encoder model."""
    msg = f"No posterior network registered for type {type(encoder)}"
    raise NotImplementedError(msg)


def posterior_network[**In, Out](
    encoder: Predictor[In, Out],
    dim: int,
    num_classes: int,
    class_counts: list | None = None,
    num_flows: int = 6,
) -> Predictor[In, Out]:
    """Create a Posterior Network predictor from an encoder based on :cite:`charpentierPosteriorNetwork2020`."""
    return posterior_network_generator(encoder, dim, num_classes, class_counts, num_flows)
