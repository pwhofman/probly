"""Shared implementation of Graph Posterior Networks."""

from __future__ import annotations

from math import log, pi
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.predictor import (
    DirichletMixtureDistributionPredictor,
    EvidentialPredictor,
)
from probly.representation.distribution import (
    DirichletDistribution,
    DirichletMixtureDistribution,
)
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


type GraphPosteriorEvidenceScale = (
    Literal[
        "latent-new",
        "latent-new-plus-classes",
        "latent-old",
        "latent-old-plus-classes",
    ]
    | None
)


@runtime_checkable
class GraphPosteriorNetworkPredictor[**In, Out: DirichletDistribution](EvidentialPredictor[In, Out], Protocol):
    """Protocol for graph posterior network predictors."""


@runtime_checkable
class LOPGraphPosteriorNetworkPredictor[**In, Out: DirichletMixtureDistribution](
    DirichletMixtureDistributionPredictor[In, Out], Protocol
):
    """Protocol for graph posterior network predictors."""


@runtime_checkable
class CUQGraphNeuralNetworkPredictor[**In, Out: DirichletDistribution](EvidentialPredictor[In, Out], Protocol):
    """Protocol for CUQ graph neural network predictors."""


def graph_evidence_log_scale(scale: GraphPosteriorEvidenceScale, latent_dim: int, num_classes: int) -> float:
    """Compute the evidence log-scale used by graph posterior networks.

    Args:
        scale: Evidence scaling variant. If ``None``, no scaling is applied.
        latent_dim: Latent feature dimensionality.
        num_classes: Number of output classes.

    Returns:
        Additive log-scale for density-derived evidence.

    Raises:
        ValueError: If ``scale`` is unknown.
    """
    if scale is None:
        return 0.0
    if scale.startswith("latent-new"):
        value = 0.5 * latent_dim * log(4 * pi)
    elif scale.startswith("latent-old"):
        value = 0.5 * (latent_dim * log(2 * pi) + log(latent_dim + 1))
    else:
        msg = f"Unknown graph posterior evidence scale: {scale!r}."
        raise ValueError(msg)
    if scale.endswith("plus-classes"):
        value += log(num_classes)
    return value


@flexdispatch
def graph_posterior_network_generator[**In, Out: DirichletDistribution](
    input_encoder: Predictor[In, Out],
    latent_dim: int,
    num_classes: int,
    *,
    encoder_dim: int | None = None,
    num_flows: int = 6,
    class_counts: list | None = None,
    evidence_scale: GraphPosteriorEvidenceScale = "latent-new",
    propagation_steps: int = 10,
    teleport_probability: float = 0.1,
    add_self_loops: bool = True,
) -> GraphPosteriorNetworkPredictor[In, Out]:
    """Return a graph posterior network given an encoder model."""
    msg = f"No graph posterior network registered for type {type(input_encoder)}"
    raise NotImplementedError(msg)


@flexdispatch
def lop_graph_posterior_network_generator[**In, Out: DirichletMixtureDistribution](
    input_encoder: Predictor[In, Out],
    latent_dim: int,
    num_classes: int,
    *,
    encoder_dim: int | None = None,
    num_flows: int = 6,
    class_counts: list | None = None,
    evidence_scale: GraphPosteriorEvidenceScale = "latent-new",
    propagation_steps: int = 10,
    teleport_probability: float = 0.1,
    add_self_loops: bool = True,
) -> LOPGraphPosteriorNetworkPredictor[In, Out]:
    """Return a LOP graph posterior network given an encoder model."""
    msg = f"No LOP graph posterior network registered for type {type(input_encoder)}"
    raise NotImplementedError(msg)


@flexdispatch
def cuq_graph_neural_network_generator[**In, Out: DirichletDistribution](
    input_encoder: Predictor[In, Out],
    latent_dim: int,
    num_classes: int,
    *,
    encoder_dim: int | None = None,
    num_flows: int = 6,
    class_counts: list | None = None,
    evidence_scale: GraphPosteriorEvidenceScale = "latent-new",
    propagation_steps: int = 10,
    teleport_probability: float = 0.1,
    add_self_loops: bool = True,
    convolution_name: Literal["appnp", "gcn"] = "appnp",
) -> CUQGraphNeuralNetworkPredictor[In, Out]:
    """Return a CUQ graph neural network given an encoder model."""
    msg = f"No CUQ graph neural network registered for type {type(input_encoder)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@GraphPosteriorNetworkPredictor.register_factory
def graph_posterior_network[**In, Out: DirichletDistribution](
    input_encoder: Predictor[In, Out],
    latent_dim: int,
    num_classes: int,
    *,
    encoder_dim: int | None = None,
    num_flows: int = 6,
    class_counts: list | None = None,
    evidence_scale: GraphPosteriorEvidenceScale = "latent-new",
    propagation_steps: int = 10,
    teleport_probability: float = 0.1,
    add_self_loops: bool = True,
) -> GraphPosteriorNetworkPredictor[In, Out]:
    """Create a Graph Posterior Network predictor.

    Args:
        input_encoder: Predictor applied to ``data.x`` before density modeling.
        latent_dim: Latent dimensionality used by the normalizing flow.
        num_classes: Number of output classes.
        encoder_dim: Output dimensionality of ``input_encoder``. Inferred when omitted.
        num_flows: Number of radial flow layers per class.
        class_counts: Optional class-count prior. If omitted, counts are inferred from ``data.train_mask``.
        evidence_scale: Additive log-scale for feature evidence.
        propagation_steps: Number of APPNP propagation steps.
        teleport_probability: APPNP teleport probability.
        add_self_loops: Whether APPNP should add self-loops.

    Returns:
        A graph posterior network predictor returning Dirichlet alphas.
    """
    return graph_posterior_network_generator(
        input_encoder,
        latent_dim,
        num_classes,
        encoder_dim=encoder_dim,
        num_flows=num_flows,
        class_counts=class_counts,
        evidence_scale=evidence_scale,
        propagation_steps=propagation_steps,
        teleport_probability=teleport_probability,
        add_self_loops=add_self_loops,
    )


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@GraphPosteriorNetworkPredictor.register_factory
def lop_graph_posterior_network[**In, Out: DirichletMixtureDistribution](
    input_encoder: Predictor[In, Out],
    latent_dim: int,
    num_classes: int,
    *,
    encoder_dim: int | None = None,
    num_flows: int = 6,
    class_counts: list | None = None,
    evidence_scale: GraphPosteriorEvidenceScale = "latent-new",
    propagation_steps: int = 10,
    teleport_probability: float = 0.1,
    add_self_loops: bool = True,
) -> LOPGraphPosteriorNetworkPredictor[In, Out]:
    """Create a LOP-GPN predictor with approximate pooled Dirichlet outputs.

    Args:
        input_encoder: Predictor applied to ``data.x`` before density modeling.
        latent_dim: Latent dimensionality used by the normalizing flow.
        num_classes: Number of output classes.
        encoder_dim: Output dimensionality of ``input_encoder``. Inferred when omitted.
        num_flows: Number of radial flow layers per class.
        class_counts: Optional class-count prior. If omitted, counts are inferred from ``data.train_mask``.
        evidence_scale: Additive log-scale for feature evidence.
        propagation_steps: Number of APPNP propagation steps.
        teleport_probability: APPNP teleport probability.
        add_self_loops: Whether APPNP should add self-loops.

    Returns:
        A LOP-GPN predictor returning approximate Dirichlet alphas.
    """
    return lop_graph_posterior_network_generator(
        input_encoder,
        latent_dim,
        num_classes,
        encoder_dim=encoder_dim,
        num_flows=num_flows,
        class_counts=class_counts,
        evidence_scale=evidence_scale,
        propagation_steps=propagation_steps,
        teleport_probability=teleport_probability,
        add_self_loops=add_self_loops,
    )


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@GraphPosteriorNetworkPredictor.register_factory
def cuq_graph_neural_network[**In, Out: DirichletDistribution](
    input_encoder: Predictor[In, Out],
    latent_dim: int,
    num_classes: int,
    *,
    encoder_dim: int | None = None,
    num_flows: int = 6,
    class_counts: list | None = None,
    evidence_scale: GraphPosteriorEvidenceScale = "latent-new",
    propagation_steps: int = 10,
    teleport_probability: float = 0.1,
    add_self_loops: bool = True,
    convolution_name: Literal["appnp", "gcn"] = "appnp",
) -> CUQGraphNeuralNetworkPredictor[In, Out]:
    """Create a CUQ-GNN predictor with graph-refined node features.

    Args:
        input_encoder: Predictor applied to ``data.x`` before graph refinement.
        latent_dim: Latent dimensionality used by the normalizing flow.
        num_classes: Number of output classes.
        encoder_dim: Output dimensionality of ``input_encoder``. Inferred when omitted.
        num_flows: Number of radial flow layers per class.
        class_counts: Optional class-count prior. If omitted, counts are inferred from ``data.train_mask``.
        evidence_scale: Additive log-scale for feature evidence.
        propagation_steps: Number of APPNP propagation steps.
        teleport_probability: APPNP teleport probability.
        add_self_loops: Whether APPNP should add self-loops.
        convolution_name: Graph convolution used before density modeling. Supports ``"appnp"`` and ``"gcn"``.

    Returns:
        A CUQ-GNN predictor returning Dirichlet alphas.
    """
    return cuq_graph_neural_network_generator(
        input_encoder,
        latent_dim,
        num_classes,
        encoder_dim=encoder_dim,
        num_flows=num_flows,
        class_counts=class_counts,
        evidence_scale=evidence_scale,
        propagation_steps=propagation_steps,
        teleport_probability=teleport_probability,
        add_self_loops=add_self_loops,
        convolution_name=convolution_name,
    )
