"""Module for graph posterior network implementations."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    GraphPosteriorNetworkPredictor,
    cuq_graph_neural_network,
    cuq_graph_neural_network_generator,
    graph_evidence_log_scale,
    graph_posterior_network,
    graph_posterior_network_generator,
    lop_graph_posterior_network,
    lop_graph_posterior_network_generator,
)


@graph_posterior_network_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@lop_graph_posterior_network_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@cuq_graph_neural_network_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "GraphPosteriorNetworkPredictor",
    "cuq_graph_neural_network",
    "graph_evidence_log_scale",
    "graph_posterior_network",
    "lop_graph_posterior_network",
]
