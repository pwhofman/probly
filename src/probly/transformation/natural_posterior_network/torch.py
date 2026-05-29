"""Torch implementation of the natural posterior network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from probly.layers.torch import RadialNormalizingFlowStack
from probly.transformation.natural_posterior_network import NaturalPosteriorNetworkPredictor
from probly.traverse_nn.utils import get_output_dim

from ._common import CertaintyBudget, budget_log_scale, natural_posterior_network_generator


@natural_posterior_network_generator.register(nn.Module)
class TorchNaturalPosteriorNetwork(nn.Module, NaturalPosteriorNetworkPredictor):
    """Torch implementation of the Natural Posterior Network."""

    alpha_prior: torch.Tensor
    """Buffer holding the per-class Dirichlet prior parameters."""

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_classes: int,
        num_flows: int = 8,
        certainty_budget: CertaintyBudget = "normal",
        alpha_prior: float = 1.0,
    ) -> None:
        """Initialize a natural posterior network."""
        super().__init__()
        self.encoder = encoder
        encoder_dim = get_output_dim(encoder)
        self.fc = nn.Linear(encoder_dim, latent_dim)
        self.batch_norm = nn.BatchNorm1d(latent_dim)
        self.norm_flow = RadialNormalizingFlowStack(dim=latent_dim, num_classes=1, num_flows=num_flows)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.certainty_budget = certainty_budget
        self.log_scale = budget_log_scale(certainty_budget, latent_dim)
        self.register_buffer("alpha_prior", torch.full((num_classes,), float(alpha_prior)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Dirichlet concentration parameters."""
        z = self.batch_norm(self.fc(self.encoder(x)))
        log_pz = self.norm_flow.log_prob(z).squeeze(-1)
        log_evidence = (log_pz + self.log_scale).clamp(-30.0, 30.0)
        log_chi = F.log_softmax(self.classifier(z), dim=-1)
        # Cast to fp32: under AMP, exp() of a moderately negative log-evidence
        # underflows in fp16 and kills the learning signal (same fix as PostNet).
        n = torch.exp(log_evidence.float())
        return self.alpha_prior + n.unsqueeze(-1) * log_chi.exp().float()
