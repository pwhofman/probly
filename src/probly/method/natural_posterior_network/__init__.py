"""Natural posterior network method compatibility exports."""

from __future__ import annotations

from probly.transformation.natural_posterior_network import (
    CertaintyBudget,
    NaturalPosteriorNetworkPredictor,
    natural_posterior_network,
)

from ._common import (
    NaturalPosteriorDecomposition,
    NaturalPosteriorNetworkRepresentation,
    NaturalPosteriorNetworkRepresenter,
)

__all__ = [
    "CertaintyBudget",
    "NaturalPosteriorDecomposition",
    "NaturalPosteriorNetworkPredictor",
    "NaturalPosteriorNetworkRepresentation",
    "NaturalPosteriorNetworkRepresenter",
    "natural_posterior_network",
]
