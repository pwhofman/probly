"""Torch Geometric implementation of Graph Posterior Networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, cast

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import APPNP, GCNConv

from probly.layers.torch import RadialNormalizingFlowStack
from probly.representation.distribution._common import DirichletMixtureDistribution
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution
from probly.representation.distribution.torch_mixture import TorchMixtureDistribution
from probly.traverse_nn.utils import get_output_dim

from ._common import (
    CUQGraphNeuralNetworkPredictor,
    GraphPosteriorEvidenceScale,
    GraphPosteriorNetworkPredictor,
    LOPGraphPosteriorNetworkPredictor,
    cuq_graph_neural_network_generator,
    graph_evidence_log_scale,
    graph_posterior_network_generator,
    lop_graph_posterior_network_generator,
)


class TorchGraphPosteriorNetworkBase[T, F](nn.Module, ABC):
    """Torch Geometric Graph Posterior Network."""

    class_counts: torch.Tensor | None

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_classes: int,
        *,
        encoder_dim: int | None = None,
        num_flows: int = 6,
        class_counts: list | torch.Tensor | None = None,
        evidence_scale: GraphPosteriorEvidenceScale = "latent-new",
        propagation_steps: int = 10,
        teleport_probability: float = 0.1,
        add_self_loops: bool = True,
    ) -> None:
        """Initialize a Graph Posterior Network.

        Args:
            encoder: Module applied to node features ``data.x``.
            latent_dim: Latent dimensionality used by the normalizing flow.
            num_classes: Number of output classes.
            encoder_dim: Output dimension of ``encoder``. Inferred when omitted.
            num_flows: Number of radial flow layers per class.
            class_counts: Optional class-count prior.
            evidence_scale: Additive log-scale for feature evidence.
            propagation_steps: Number of APPNP propagation steps.
            teleport_probability: APPNP teleport probability.
            add_self_loops: Whether APPNP should add self-loops.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.encoder_dim = encoder_dim if encoder_dim is not None else get_output_dim(encoder)
        self.latent_encoder = nn.Linear(self.encoder_dim, latent_dim)
        self.batch_norm = nn.BatchNorm1d(latent_dim)
        self.norm_flow = RadialNormalizingFlowStack(dim=latent_dim, num_classes=num_classes, num_flows=num_flows)
        self.propagation = APPNP(K=propagation_steps, alpha=teleport_probability, add_self_loops=add_self_loops)
        self.register_buffer(
            "class_counts",
            None if class_counts is None else torch.as_tensor(class_counts, dtype=torch.float),
        )
        self.evidence_log_scale = graph_evidence_log_scale(evidence_scale, latent_dim, num_classes)

    def _validate_data(self, data: Data) -> None:
        if not isinstance(data, Data) or data.x is None or data.edge_index is None:
            msg = "Graph Posterior Networks expect a torch_geometric.data.Data object with x and edge_index."
            raise TypeError(msg)

    def encode(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode graph node features into hidden and latent representations.

        Args:
            data: PyG data object with ``x`` and ``edge_index``.

        Returns:
            Hidden and latent node representations.
        """
        self._validate_data(data)
        hidden = self.encoder(data.x)
        latent = self.latent_encoder(hidden)
        latent = self.batch_norm(latent)
        return hidden, latent

    def class_probabilities(self, data: Data) -> torch.Tensor:
        """Compute class-prior probabilities from configured or training labels.

        Args:
            data: PyG data object with labels and ``train_mask`` when no fixed prior is configured.

        Returns:
            Class probabilities with shape ``(num_classes,)``.
        """
        x = cast("torch.Tensor", data.x)
        if self.class_counts is not None:
            counts = self.class_counts.to(device=x.device, dtype=torch.float)
        else:
            if getattr(data, "y", None) is None or getattr(data, "train_mask", None) is None:
                msg = "data.y and data.train_mask are required when class_counts is omitted."
                raise ValueError(msg)
            y = cast("torch.Tensor", data.y)
            train_mask = cast("torch.Tensor", data.train_mask)
            labels = y[train_mask]
            counts = torch.bincount(labels, minlength=self.num_classes).to(device=x.device, dtype=torch.float)
        counts = counts.clamp_min(1e-6)
        return counts / counts.sum()

    def _feature_evidence_from_latent(self, data: Data, latent: torch.Tensor) -> torch.Tensor:
        log_density = self.norm_flow.log_prob(latent)
        log_class_prior = self.class_probabilities(data).log()
        log_beta = (log_density + log_class_prior + self.evidence_log_scale).clamp(-30.0, 30.0)
        return torch.exp(log_beta.float())

    def feature_evidence(self, data: Data) -> dict[str, torch.Tensor]:
        """Compute feature-level evidence terms before graph propagation.

        Args:
            data: PyG data object with ``x`` and ``edge_index``.

        Returns:
            Dictionary containing hidden, latent, and feature evidence tensors.
        """
        hidden, latent = self.encode(data)
        beta_ft = self._feature_evidence_from_latent(data, latent)
        return {"hidden": hidden, "latent": latent, "beta_ft": beta_ft}

    def _refine_hidden(self, hidden: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        return hidden

    def forward(self, data: Data) -> F:
        """Compute graph posterior Dirichlet alphas for all nodes."""
        self._validate_data(data)
        x: torch.Tensor = data.x  # ty:ignore[invalid-assignment]
        edge_index: torch.Tensor = data.edge_index  # ty:ignore[invalid-assignment]
        hidden = self.encoder(x)
        hidden = self._refine_hidden(hidden, edge_index)
        latent = self.latent_encoder(hidden)
        beta_ft = self._feature_evidence_from_latent(data, latent)
        beta = self.propagation(beta_ft, edge_index)
        return 1.0 + beta

    @abstractmethod
    def predict_representation(self, data: Data) -> T:
        """Compute the predictive distribution parameters for all nodes."""


@graph_posterior_network_generator.register(nn.Module)
class TorchGraphPosteriorNetwork(
    TorchGraphPosteriorNetworkBase[TorchDirichletDistribution, torch.Tensor],
    GraphPosteriorNetworkPredictor[[Data], TorchDirichletDistribution],
):
    """Torch Geometric Graph Posterior Network."""

    def predict_representation(self, data: Data) -> TorchDirichletDistribution:
        """Compute the predictive distribution parameters (Dirichlet alphas) for all nodes."""
        alphas: torch.Tensor = self.forward(data)
        return TorchDirichletDistribution(alphas=alphas)


@lop_graph_posterior_network_generator.register(nn.Module)
class TorchLOPGraphPosteriorNetwork(
    TorchGraphPosteriorNetworkBase[
        TorchMixtureDistribution[TorchDirichletDistribution, TorchCategoricalDistribution],
        tuple[torch.Tensor, torch.Tensor],
    ],
    LOPGraphPosteriorNetworkPredictor[
        [Data], DirichletMixtureDistribution[TorchDirichletDistribution, TorchCategoricalDistribution]
    ],
):
    """Torch Geometric LOP-GPN with approximate pooled Dirichlet outputs."""

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute raw LOP-GPN terms including dense propagation weights.

        Args:
            data: PyG data object with ``x`` and ``edge_index``.

        Returns:
            Raw output with approximate alphas and feature-level mixture terms.
        """
        self._validate_data(data)
        x: torch.Tensor = data.x  # ty:ignore[invalid-assignment]
        edge_index: torch.Tensor = data.edge_index  # ty:ignore[invalid-assignment]
        hidden = self.encoder(x)
        latent = self.latent_encoder(hidden)
        beta_ft = self._feature_evidence_from_latent(data, latent)
        alpha_features = 1.0 + beta_ft
        num_nodes = data.num_nodes
        if num_nodes is None:
            msg = "data.num_nodes is required for LOP-GPN forward_raw."
            raise ValueError(msg)
        identity = torch.eye(num_nodes, device=beta_ft.device, dtype=beta_ft.dtype)
        mixture_weights = self.propagation(identity, edge_index)

        return alpha_features, mixture_weights

    def predict_representation(
        self, data: Data
    ) -> TorchMixtureDistribution[TorchDirichletDistribution, TorchCategoricalDistribution]:
        """Compute the predictive distribution parameters (Dirichlet alphas) for all nodes."""
        alphas, weights = self.forward(data)
        return TorchMixtureDistribution(
            TorchDirichletDistribution(alphas=alphas.expand(weights.shape[0], *alphas.shape)),
            weights,
        )


@cuq_graph_neural_network_generator.register(nn.Module)
class TorchCUQGraphNeuralNetwork(
    TorchGraphPosteriorNetworkBase[TorchDirichletDistribution, torch.Tensor],
    CUQGraphNeuralNetworkPredictor[[Data], TorchDirichletDistribution],
):
    """Torch Geometric CUQ-GNN using graph-refined hidden features."""

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_classes: int,
        *,
        encoder_dim: int | None = None,
        num_flows: int = 6,
        class_counts: list | torch.Tensor | None = None,
        evidence_scale: GraphPosteriorEvidenceScale = "latent-new",
        propagation_steps: int = 10,
        teleport_probability: float = 0.1,
        add_self_loops: bool = True,
        convolution_name: Literal["appnp", "gcn"] = "appnp",
    ) -> None:
        """Initialize a CUQ-GNN.

        Args:
            encoder: Module applied to node features ``data.x``.
            latent_dim: Latent dimensionality used by the normalizing flow.
            num_classes: Number of output classes.
            encoder_dim: Output dimension of ``encoder``. Inferred when omitted.
            num_flows: Number of radial flow layers per class.
            class_counts: Optional class-count prior.
            evidence_scale: Additive log-scale for feature evidence.
            propagation_steps: Number of APPNP propagation steps.
            teleport_probability: APPNP teleport probability.
            add_self_loops: Whether APPNP should add self-loops.
            convolution_name: Hidden-feature graph module, either ``"appnp"`` or ``"gcn"``.
        """
        super().__init__(
            encoder,
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
        if convolution_name == "appnp":
            self.graph_module = APPNP(K=propagation_steps, alpha=teleport_probability, add_self_loops=add_self_loops)
        elif convolution_name == "gcn":
            self.graph_module = GCNConv(self.encoder_dim, self.encoder_dim)
        else:
            msg = f"Unsupported CUQ-GNN convolution: {convolution_name!r}."
            raise ValueError(msg)
        self.convolution_name = convolution_name

    def _refine_hidden(self, hidden: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.convolution_name == "gcn":
            return self.graph_module(hidden, edge_index)
        return self.graph_module(hidden, edge_index)

    def predict_representation(self, data: Data) -> TorchDirichletDistribution:
        """Compute the predictive distribution parameters (Dirichlet alphas) for all nodes."""
        alphas: torch.Tensor = self.forward(data)
        return TorchDirichletDistribution(alphas=alphas)
