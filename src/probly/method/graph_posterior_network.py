"""Graph posterior network method compatibility exports."""

from __future__ import annotations

from probly.transformation.graph_posterior_network import (
    GraphPosteriorNetworkPredictor,
    cuq_graph_neural_network,
    graph_posterior_network,
    lop_graph_posterior_network,
)

__all__ = [
    "GraphPosteriorNetworkPredictor",
    "cuq_graph_neural_network",
    "graph_posterior_network",
    "lop_graph_posterior_network",
]
