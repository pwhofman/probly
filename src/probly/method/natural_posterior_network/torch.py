"""Torch implementation of the natural posterior network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from probly.layers.torch import RadialNormalizingFlowStack
from probly.traverse_nn.utils import get_output_dim

from ._common import CertaintyBudget, budget_log_scale, natural_posterior_network_generator


@natural_posterior_network_generator.register(nn.Module)
class TorchNaturalPosteriorNetwork(nn.Module):
    """Torch implementation of the Natural Posterior Network.

    Computes a Bayesian posterior update over a Dirichlet distribution per
    sample. The encoder is projected to a low-dim latent ``z``; a single
    shared normalizing flow yields ``log p(z)``; a small linear classifier
    on the latent yields class log-probabilities ``log chi(x)``. The
    Dirichlet parameters are returned as
    ``alpha = alpha_prior + n(x) * chi(x)``, where ``n(x)`` is the
    budget-scaled, clamped exponential of ``log p(z)``.

    Attributes:
        encoder: User-supplied feature encoder.
        fc: Projection from encoder features to the latent space.
        batch_norm: Per-feature normalization of the latent before the flow.
        norm_flow: Single shared normalizing flow modelling ``p(z)``.
        classifier: Linear classifier producing class logits from the latent.
        alpha_prior: Buffer holding the per-class Dirichlet prior parameters.
        certainty_budget: The selected budget scheme.
        log_scale: Additive constant applied to ``log p(z)`` per the budget.
    """

    alpha_prior: torch.Tensor

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_classes: int,
        num_flows: int = 8,
        certainty_budget: CertaintyBudget = "normal",
        alpha_prior: float = 1.0,
    ) -> None:
        """Initialize a natural posterior network.

        Args:
            encoder: Feature encoder mapping inputs to a feature tensor of
                shape ``(B, encoder_dim)``.
            latent_dim: Dimension ``H`` of the latent space.
            num_classes: Number of output classes ``K``.
            num_flows: Number of radial flow layers in the shared flow.
            certainty_budget: Named scheme for scaling ``log p(z)`` before
                exponentiation. Defaults to ``"normal"``.
            alpha_prior: Per-class Dirichlet prior parameter (uniform across
                classes). Defaults to ``1.0``.
        """
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
        """Forward pass returning Dirichlet concentration parameters.

        Args:
            x: Input tensor consumed by the encoder.

        Returns:
            Dirichlet concentration parameters of shape ``(B, num_classes)``,
            all strictly positive.
        """
        z = self.batch_norm(self.fc(self.encoder(x)))
        log_pz = self.norm_flow.log_prob(z).squeeze(-1)
        log_evidence = (log_pz + self.log_scale).clamp(-30.0, 30.0)
        log_chi = F.log_softmax(self.classifier(z), dim=-1)
        # Cast to fp32: under AMP, exp() of a moderately negative log-evidence
        # underflows in fp16 and kills the learning signal (same fix as PostNet).
        n = torch.exp(log_evidence.float())
        return self.alpha_prior + n.unsqueeze(-1) * log_chi.exp().float()
