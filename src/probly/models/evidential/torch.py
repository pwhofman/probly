"""Models for evidential deep learning using PyTorch."""

from __future__ import annotations

import torch
from torch import nn

import probly.layers.evidential.torch as t


class NatPN(nn.Module):
    """Natural Posterior Network for evidential deep learning with normalizing flows.

    Combines encoder, normalizing flow density, and head for uncertainty quantification.
    Users can provide custom encoders for different data modalities.

    Args:
        encoder: Encoder module mapping raw inputs to latent space.
        head: Head module (DirichletHead for classification, GaussianHead for regression).
        latent_dim: Dimension of the latent space.
        flow_length: Number of radial flow layers. Defaults to 4.
        certainty_budget: Budget for certainty calibration. If None, defaults to latent_dim.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module | None = None,
        latent_dim: int = 2,
        flow_length: int = 4,
        certainty_budget: float = 2.0,
    ) -> None:
        """Initialize the NatPN model."""
        super().__init__()

        if head is None:
            head = t.DirichletHead(latent_dim=latent_dim, num_classes=10)

        self.encoder = encoder
        self.head = head

        self.flow = t.RadialFlowDensity(
            dim=latent_dim,
            flow_length=flow_length,
        )

        if certainty_budget is None:
            certainty_budget = float(latent_dim)
        self.certainty_budget = certainty_budget

    def freeze_encoder(self) -> None:
        """Freeze encoder weights (for transfer learning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights (for fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through encoder, flow, and head.

        Args:
            x: Input tensor compatible with the encoder.

        Returns:
            Dictionary with predictions from the head (including alpha for classification,
            or mean/var for regression) along with latent space information.
        """
        z = self.encoder(x)  # [B, latent_dim]
        log_pz = self.flow.log_prob(z)  # [B]

        return self.head(
            z=z,
            log_pz=log_pz,
            certainty_budget=self.certainty_budget,
        )


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
            encoder = t.EncoderMnist(latent_dim=latent_dim)

        if head is None:
            head = t.SimpleHead(latent_dim=latent_dim, num_classes=num_classes)

        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        z = self.encoder(x)
        return self.head(z)


class EvidentialRegressionModel(nn.Module):
    """Full evidential regression model combining encoder and evidential head."""

    def __init__(self, encoder: nn.Module | None = None) -> None:
        """Initialize the full model."""
        super().__init__()
        if encoder is None:
            encoder = t.MLPEncoder(feature_dim=32)

        self.encoder = encoder
        self.head = t.EvidentialHead(feature_dim=encoder.feature_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and head."""
        features = self.encoder(x)
        return self.head(features)


class DirichletClassificationModel(nn.Module):
    """Full model combining encoder and Dirichlet head for evidential classification.

    This model learns to output Dirichlet concentration parameters (alpha)
    for each input, enabling uncertainty quantification via the Dirichlet distribution.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
    ) -> None:
        """Initialize the full Dirichlet classification model.

        Args:
            input_dim: Size of input features (flattened).
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension for encoder (default: 128).
            latent_dim: Latent dimension for encoder (default: 128).
        """
        super().__init__()
        self.encoder = t.DirichletMLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.head = t.IRDHead(latent_dim=latent_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and head.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Alpha parameters of shape (batch_size, num_classes).
        """
        features = self.encoder(x)
        alpha = self.head(features)
        return alpha


class PostNetModel(nn.Module):
    """Posterior Network model containing encoder + flow."""

    def __init__(
        self,
        encoder: nn.Module | None = None,
        flow: nn.Module | None = None,
        input_dim: int = 784,
        hidden_dim: int = 256,
        latent_dim: int = 6,
    ) -> None:
        """Initialize a Posterior Network model."""
        super().__init__()

        if encoder is None:
            encoder = t.FlattenMLPEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

        if flow is None:
            flow = t.BatchedRadialFlowDensity(num_classes=10, dim=latent_dim, flow_length=6)

        self.encoder = encoder
        self.flow = flow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs into latent representations.

        Args:
            x: Input tensor of shape (B, ...) where B is batch size.

        Returns:
            Latent tensor z of shape (B, latent_dim).
        """
        return self.encoder(x)
