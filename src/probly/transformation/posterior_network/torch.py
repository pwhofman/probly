"""Torch implementation of the posterior network."""

from __future__ import annotations

import torch
from torch import nn

from probly.layers.torch import RadialNormalizingFlowStack
from probly.transformation.posterior_network import PosteriorNetworkPredictor
from probly.traverse_nn.utils import get_output_dim

from ._common import posterior_network_generator


@posterior_network_generator.register(nn.Module)
class TorchPosteriorNetwork(nn.Module, PosteriorNetworkPredictor):
    """Torch version of Posterior Network."""

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_classes: int,
        *,
        encoder_dim: int | None = None,
        class_counts: list | torch.Tensor | None = None,
        num_flows: int = 6,
    ) -> None:
        """Initialize a posterior network."""
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = get_output_dim(encoder) if encoder_dim is None else encoder_dim
        self.latent_encoder = nn.Linear(self.encoder_dim, latent_dim)
        self.batch_norm = nn.BatchNorm1d(latent_dim)
        self.norm_flow = RadialNormalizingFlowStack(dim=latent_dim, num_classes=num_classes, num_flows=num_flows)
        self.register_buffer("class_counts", torch.as_tensor(class_counts, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of posterior network."""
        x = self.encoder(x)
        x = self.latent_encoder(x)
        x = self.batch_norm(x)
        log_density = self.norm_flow.log_prob(x)
        # Compute alphas in fp32: under AMP, exp() of a moderately negative
        # log_density underflows to 0 in fp16 and kills the learning signal.
        alphas = 1.0 + torch.exp(log_density.float()) * self.class_counts.float()
        return alphas
