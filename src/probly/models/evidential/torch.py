"""Collection of torch evidential models."""

from __future__ import annotations

import torch
from torch import nn

from probly.layers.evidential.torch import Encoder, RadialFlowDensity


class NatPNClassifier(nn.Module):
    """Natural Posterior Network for classification with a Dirichlet posterior over class probabilities."""

    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 2,
        flow_length: int = 4,
        certainty_budget: float | None = None,
        n_prior: float | None = None,
    ) -> None:
        """Initialize the NatPN classifier and its components."""
        super().__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # 1. Encoder: x -> z
        self.encoder = Encoder(latent_dim=latent_dim)

        # 2. Decoder: z -> logits for each class (SMOOTHED)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        # 3. Single normalizing flow density over z
        self.flow = RadialFlowDensity(dim=latent_dim, flow_length=flow_length)

        # 4. Certainty budget N_H: scales the density into "evidence"
        #    Intuition: total evidence mass to distribute over the latent space.
        if certainty_budget is None:
            certainty_budget = float(latent_dim)
        self.certainty_budget = certainty_budget

        # 5. Prior pseudo-count n_prior and prior χ_prior
        if n_prior is None:
            n_prior = float(num_classes)

        # χ_prior: uniform over classes
        chi_prior = torch.full((num_classes,), 1.0 / num_classes)  # [C]
        alpha_prior = n_prior * chi_prior  # [C] -> Dirichlet(1,1,...,1)

        # Register as buffer so it is moved automatically with model.to(device)
        self.register_buffer("alpha_prior", alpha_prior)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input batch, shape [B, 1, 28, 28] for MNIST.

        Returns:
            alpha: Posterior Dirichlet parameters, shape [B, C].
            z: Latent representation, shape [B, latent_dim].
            log_pz: Log-density log p(z) under the flow, shape [B].
        """
        # Encode to latent space
        z = self.encoder(x)  # [B, latent_dim]

        # Class logits -> per-class χ (like normalized preferences)
        logits = self.classifier(z)  # [B, C]
        chi = torch.softmax(logits, dim=-1)  # [B, C], sums to 1

        # Flow density over z -> log p(z)
        log_pz = self.flow.log_prob(z)  # [B]

        # Convert density into scalar evidence n(x)
        # n(x) = N_H * exp(log p(z)) = N_H * p(z)
        n = self.certainty_budget * log_pz.exp()  # [B], evidence ≥ 0
        n = torch.clamp(n, min=1e-8)  # avoid exact zero for numerical stability

        # Evidence per class: n_i * χ_i  -> pseudo-counts
        evidence = n.unsqueeze(-1) * chi  # [B, C]

        # Posterior Dirichlet parameters: alpha = alpha_prior + evidence
        alpha = self.alpha_prior.unsqueeze(0) + evidence  # [B, C]

        return alpha, z, log_pz
