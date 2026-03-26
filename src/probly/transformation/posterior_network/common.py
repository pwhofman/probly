"""Shared implementation of Posterior Networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.transformation.evidential import evidential_classification

if TYPE_CHECKING:
    from probly.predictor import Predictor


def posterior_network[**In, Out](base: Predictor[In, Out]) -> Predictor[In, Out]:
    """Create a Posterior Network predictor from a base model based on :cite:`charpentierPosteriorNetwork2020`."""
    return evidential_classification(base)
