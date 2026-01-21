"""Models for evidential deep learning using PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from probly.layers.evidential.torch import EncoderMnist, RadialFlowDensity, SimpleHead


class NatPN(nn.Module):
    """Docstring for NatPN."""

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        latent_dim: int,
        flow_length: int = 4,
        certainty_budget: float | None = None,
    ) -> None:
        """Init for NatPN model."""
        super().__init__()

        self.encoder = encoder
        self.head = head

        self.flow = RadialFlowDensity(
            dim=latent_dim,
            flow_length=flow_length,
        )

        if certainty_budget is None:
            certainty_budget = float(latent_dim)
        self.certainty_budget = certainty_budget

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Infer forward pass."""
        z = self.encoder(x)  # [B, latent_dim]
        log_pz = self.flow.log_prob(z)  # [B]

        return self.head(
            z=z,
            log_pz=log_pz,
            certainty_budget=self.certainty_budget,
        )


class DirichletHead(nn.Module):
    """Dirichlet posterior head for classification."""

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        n_prior: float | None = None,
    ) -> None:
        """Init for DirichletHead."""
        super().__init__()

        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        if n_prior is None:
            n_prior = float(num_classes)

        chi_prior = torch.full((num_classes,), 1.0 / num_classes)
        alpha_prior = n_prior * chi_prior

        self.register_buffer("alpha_prior", alpha_prior)

    def forward(
        self,
        z: torch.Tensor,
        log_pz: torch.Tensor,
        certainty_budget: float,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for DirichletHead."""
        logits = self.classifier(z)  # [B, C]
        chi = torch.softmax(logits, dim=-1)  # [B, C]

        # Total evidence n(x)
        n = certainty_budget * log_pz.exp()  # [B]
        n = torch.clamp(n, min=1e-8)

        evidence = n.unsqueeze(-1) * chi  # [B, C]
        alpha = self.alpha_prior.unsqueeze(0) + evidence

        return {
            "alpha": alpha,  # Dirichlet parameters
            "z": z,
            "log_pz": log_pz,
            "evidence": evidence,
        }


class GaussianHead(nn.Module):
    """Gaussian posterior head for regression."""

    def __init__(
        self,
        latent_dim: int,
        out_dim: int = 1,
    ) -> None:
        """Init for GaussianHead."""
        super().__init__()

        self.mean_net = nn.Linear(latent_dim, out_dim)
        self.log_var_net = nn.Linear(latent_dim, out_dim)

    def forward(
        self,
        z: torch.Tensor,
        log_pz: torch.Tensor,
        certainty_budget: float,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for GaussianHead."""
        mean = self.mean_net(z)  # [B, D]
        log_var = self.log_var_net(z)  # [B, D]

        # Epistemic uncertainty via density scaling
        precision = certainty_budget * log_pz.exp().unsqueeze(-1)
        precision = torch.clamp(precision, min=1e-8)

        var = torch.exp(log_var) / precision

        return {
            "mean": mean,
            "var": var,
            "z": z,
            "log_pz": log_pz,
            "precision": precision,
        }


class SimpleCNN(nn.Module):
    """Simple CNN model for evidential classification."""

    def __init__(  # noqa: D107
        self,
        encoder: nn.Module | None = None,
        head: nn.Module | None = None,
        latent_dim: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if encoder is None:
            encoder = EncoderMnist(latent_dim=latent_dim)

        if head is None:
            head = SimpleHead(latent_dim=latent_dim, num_classes=num_classes)

        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        z = self.encoder(x)
        return self.head(z)
