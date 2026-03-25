"""Models for evidential deep learning using PyTorch."""

from __future__ import annotations

import torch
from torch import nn

import probly.layers.evidential.torch as t


class EDLModel(nn.Module):
    """Simple model for evidential deep learning (EDL) classification.

    Combines an encoder with an evidential prediction head to model
    classification uncertainty following Sensoy et al. (2018).

    Reference:
        Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty",
        NeurIPS 2018.
        https://arxiv.org/abs/1806.01768
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module | None = None,
        latent_dim: int | None = None,
        num_classes: int = 10,
    ) -> None:
        """Initialize the EDLModel for evidential classification.

        Args:
            encoder: Encoder module mapping inputs to latent space.
            head: Head module for evidential output (defaults to EDLHead).
            latent_dim: Dimension of the latent space.
            num_classes: Number of output classes.
        """
        super().__init__()

        if latent_dim is None:
            latent_dim = getattr(encoder, "latent_dim", None)
            if not isinstance(latent_dim, int):
                msg = "latent_dim must be specified if encoder has no attribute 'latent_dim'"
                raise ValueError(msg)

        if head is None:
            head = t.EDLHead(latent_dim=latent_dim, num_classes=num_classes)

        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and head.

        Args:
            x: Input tensor compatible with the encoder.

        Returns:
            Output tensor from the head module.
        """
        features = self.encoder(x)
        return self.head(features)


class PrNetModel(nn.Module):
    """Dirichlet Prior Network model for evidential classification.

    Combines an encoder with a Dirichlet Prior Network head to model
    distributional uncertainty following Malinin and Gales (2018).

    Reference:
        Malinin and Gales, "Predictive Uncertainty Estimation via Prior Networks",
        NeurIPS 2018.
        https://arxiv.org/abs/1802.10501
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module | None = None,
        latent_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        """Initialize the Dirichlet Prior Network model.

        Args:
            encoder: Encoder module mapping inputs to latent space.
            head: Dirichlet Prior Network head producing concentration parameters.
                If None, defaults to ``t.PrNetHead``.
            latent_dim: Dimensionality of the latent space.
            num_classes: Number of output classes.
        """
        super().__init__()

        if head is None:
            head = t.PrNetHead(
                latent_dim=latent_dim,
                num_classes=num_classes,
            )

        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Dirichlet parameters for input samples."""
        features = self.encoder(x)
        return self.head(features)


class PostNetModel(nn.Module):
    """Posterior Network model for evidential classification.

    Combines an encoder with class-conditional normalizing flows to model
    uncertainty-aware predictions following Malinin and Gales (2020).

    Reference:
        Malinin and Gales, "Posterior Networks: Uncertainty Estimation without
        OOD Samples via Density-Based Pseudo-Counts", NeurIPS 2020.
        https://arxiv.org/abs/2006.09239
    """

    class_counts: torch.Tensor

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int = 6,
        num_classes: int = 10,
        flow: t.BatchedRadialFlowDensity | None = None,
        class_counts: torch.Tensor | None = None,
    ) -> None:
        """Initialize a Posterior Network model.

        Args:
            encoder: Encoder mapping inputs to a latent space.
            latent_dim: Dimensionality of the latent space.
            num_classes: Number of output classes.
            flow: Class-conditional normalizing flow.
            class_counts: Empirical class counts used as a prior.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        if flow is None:
            flow = t.BatchedRadialFlowDensity(num_classes=num_classes, latent_dim=latent_dim, flow_length=6)
        self.flow = flow

        if class_counts is None:
            class_counts = torch.ones(num_classes)
        self.register_buffer("class_counts", class_counts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the Posterior Network.

        Args:
            x: Input tensor of shape (batch_size, ...) compatible with the encoder.

        Returns:
            alpha: Dirichlet concentration parameters of shape.
            p_mean: Predictive mean of the Dirichlet distribution.
            z: Latent representation of the input.
        """
        features = self.encoder(x)

        log_dens = self.flow.log_prob(features)
        dens = log_dens.exp()

        beta = dens * self.class_counts.unsqueeze(0)
        alpha = beta + 1.0
        alpha0 = alpha.sum(dim=1, keepdim=True)
        p_mean = alpha / alpha0

        return alpha, p_mean, features


class NatPNModel(nn.Module):
    """Natural Posterior Network for evidential deep learning with normalizing flows.

    Combines encoder, normalizing flow density, and head for uncertainty quantification
    following Charpentier et al., 2022. Users can provide custom encoders for different
    data modalities.

    Reference:
        Charpentier et al., "Natural Posterior Network", NeurIPS 2022.
        https://arxiv.org/abs/2105.04471
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module | None = None,
        latent_dim: int | None = None,
        flow_length: int = 4,
        certainty_budget: float = 2.0,
    ) -> None:
        """Initialize the NatPN model.

        Args:
            encoder: Encoder mapping inputs to a latent space.
            head: Prediction head operating on latent features. If None,
                a default classification head is used.
            latent_dim: Dimensionality of the latent space. If None, inferred
                from the encoder.
            flow_length: Number of radial flow layers.
            certainty_budget: Budget controlling certainty calibration.
        """
        super().__init__()

        if latent_dim is None:
            latent_dim = getattr(encoder, "latent_dim", None)
            if not isinstance(latent_dim, int):
                msg = "latent_dim must be specified if encoder has no attribute 'latent_dim'"
                raise ValueError(msg)

        if head is None:
            head = t.NatPNClassHead(latent_dim=latent_dim, num_classes=10)

        self.encoder = encoder
        self.head = head

        self.flow = t.RadialFlowDensity(
            latent_dim=latent_dim,
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
        features = self.encoder(x)  # [B, latent_dim]
        log_pz = self.flow.log_prob(features)  # [B]

        return self.head(
            features=features,
            log_pz=log_pz,
            certainty_budget=self.certainty_budget,
        )


class IRDModel(nn.Module):
    """Full model for evidential classification using Dirichlet outputs.

    Combines an encoder with a Dirichlet head to produce concentration
    parameters for uncertainty-aware classification following
    Tsiligkaridis (2019).

    Reference:
        Tsiligkaridis, "Information Robust Dirichlet Networks", 2019.
        https://arxiv.org/abs/1910.04819
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module | None = None,
        latent_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        """Initialize the IRDModel for Dirichlet-based classification.

        Args:
            encoder: Encoder module mapping inputs to latent space.
            head: Dirichlet head module mapping latent features to alpha parameters.
                If None, defaults to ``t.IRDHead``.
            latent_dim: Dimensionality of the latent space.
            num_classes: Number of output classes.
        """
        super().__init__()

        if head is None:
            head = t.IRDHead(latent_dim=latent_dim, num_classes=num_classes)

        self.encoder = encoder
        self.head = head

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


class EvidentialRegressionModel(nn.Module):
    """Full evidential regression model combining encoder and evidential head.

    Implements evidential regression for uncertainty-aware prediction, inspired by
    Deep Evidential Regression and Regression Prior Networks.

    References:
        Amini et al., "Deep Evidential Regression", NeurIPS 2020.
        https://arxiv.org/abs/1910.02600

        Malinin et al., "Regression Prior Networks", NeurIPS 2020.
        https://arxiv.org/abs/2006.11590
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module | None = None,
        latent_dim: int | None = None,
    ) -> None:
        """Initialize the EvidentialRegressionModel.

        Args:
            encoder: Encoder module mapping inputs to a latent space.
            head: Evidential regression head producing distribution parameters.
                If None, defaults to ``t.RegressionHead``.
            latent_dim: Dimensionality of the latent space. If None, inferred
                from ``encoder.latent_dim``.
        """
        super().__init__()

        if latent_dim is None:
            latent_dim = getattr(encoder, "latent_dim", None)
            if not isinstance(latent_dim, int):
                msg = "latent_dim must be specified if encoder has no attribute 'latent_dim'"
                raise ValueError(msg)

        if head is None:
            head = t.RegressionHead(latent_dim=latent_dim)

        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and head."""
        features = self.encoder(x)
        return self.head(features)
