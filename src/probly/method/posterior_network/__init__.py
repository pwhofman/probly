"""Posterior network method compatibility exports."""

from __future__ import annotations

from probly.transformation.posterior_network import PosteriorNetworkPredictor, posterior_network

from ._common import PosteriorNetworkDecomposition, PosteriorNetworkRepresentation, PosteriorNetworkRepresenter

__all__ = [
    "PosteriorNetworkDecomposition",
    "PosteriorNetworkPredictor",
    "PosteriorNetworkRepresentation",
    "PosteriorNetworkRepresenter",
    "posterior_network",
]
