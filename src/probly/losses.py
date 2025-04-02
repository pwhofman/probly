"""Collection of loss implementations."""

import torch
import torch.nn.functional as F
from torch import nn


class EvidentialLogLoss(nn.Module):
    """Evidential Log Loss based on https://arxiv.org/abs/1806.01768."""

    def __init__(self) -> None:
        """Intialize an instance of the EvidentialLogLoss class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential log loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        alphas = inputs + 1.0
        strengths = torch.sum(alphas, dim=1)
        loss = torch.mean(torch.log(strengths) - torch.log(alphas[torch.arange(targets.shape[0]), targets]))
        return loss


class EvidentialCELoss(nn.Module):
    """Evidential Cross Entropy Loss based on https://arxiv.org/abs/1806.01768."""

    def __init__(self) -> None:
        """Intialize an instance of the EvidentialCELoss class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential cross entropy loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        alphas = inputs + 1.0
        strengths = torch.sum(alphas, dim=1)
        loss = torch.mean(torch.digamma(strengths) - torch.digamma(alphas[torch.arange(targets.shape[0]), targets]))
        return loss


class EvidentialMSELoss(nn.Module):
    """Evidential Mean Square Error Loss based on https://arxiv.org/abs/1806.01768."""

    def __init__(self) -> None:
        """Intialize an instance of the EvidentialMSELoss class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential mean squared error loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        alphas = inputs + 1.0
        strengths = torch.sum(alphas, dim=1)
        y = F.one_hot(targets, inputs.shape[1])
        p = alphas / strengths[:, None]
        err = (y - p) ** 2
        var = p * (1 - p) / (strengths[:, None] + 1)
        loss = torch.mean(torch.sum(err + var, dim=1))
        return loss


class EvidentialKLDivergence(nn.Module):
    """Evidential KL Divergence Loss based on https://arxiv.org/abs/1806.01768."""

    def __init__(self) -> None:
        """Initialize an instance of the EvidentialKLDivergence class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential KL divergence loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        alphas = inputs + 1.0
        y = F.one_hot(targets, inputs.shape[1])
        alphas_tilde = y + (1 - y) * alphas
        strengths_tilde = torch.sum(alphas_tilde, dim=1)
        k = torch.full((inputs.shape[0],), inputs.shape[1], device=inputs.device)
        first = torch.lgamma(strengths_tilde) - torch.lgamma(k) - torch.sum(torch.lgamma(alphas_tilde), dim=1)
        second = torch.sum(
            (alphas_tilde - 1) * (torch.digamma(alphas_tilde) - torch.digamma(strengths_tilde)[:, None]),
            dim=1,
        )
        loss = torch.mean(first + second)
        return loss


class EvidentialNIGNLLLoss(nn.Module):
    """Evidential normal inverse gamma negative log likelihood loss.

    Implementation is based on https://arxiv.org/abs/1910.02600.
    """

    def __init__(self) -> None:
        """Intializes an instance of the EvidentialNIGNLLLoss class."""
        super().__init__()

    def forward(self, inputs: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential normal inverse gamma negative log likelihood loss.

        Args:
            inputs: dict[str, torch.Tensor] with keys 'gamma', 'nu', 'alpha', 'beta'
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        omega = 2 * inputs["beta"] * (1 + inputs["nu"])
        loss = (
            0.5 * torch.log(torch.pi / inputs["nu"])
            - inputs["alpha"] * torch.log(omega)
            + (inputs["alpha"] + 0.5) * torch.log((targets - inputs["gamma"]) ** 2 * inputs["nu"] + omega)
            + torch.lgamma(inputs["alpha"])
            - torch.lgamma(inputs["alpha"] + 0.5)
        ).mean()
        return loss


class EvidentialRegressionRegularization(nn.Module):
    """Implementation of the evidential regression regularization."""

    def __init__(self) -> None:
        """Initialize an instance of the evidential regression regularization class."""
        super().__init__()

    def forward(self, inputs: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential regression regularization.

        Args:
            inputs: dict[str, torch.Tensor] with keys 'gamma', 'nu', 'alpha', 'beta'
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        loss = (torch.abs(targets - inputs["gamma"]) * (2 * inputs["nu"] + inputs["alpha"])).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss based on https://arxiv.org/abs/1708.02002.

    Attributes:
        alpha: float, control importance of minority class
        gamma: float, control loss for hard instances
    """

    def __init__(self, alpha: float = 1, gamma: float = 2) -> None:
        """Initializes an instance of the FocalLoss class.

        Args:
            alpha: float, control importance of minority class
            gamma: float, control loss for hard instances
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the focal loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])
        prob = F.softmax(inputs, dim=-1)
        p_t = torch.sum(prob * targets_one_hot, dim=-1)

        log_prob = torch.log(prob)
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.sum(log_prob * targets_one_hot, dim=-1)

        return torch.mean(loss)


class ELBOLoss(nn.Module):
    """Evidence lower bound loss based on https://arxiv.org/abs/1505.05424.

    Attributes:
        kl_penalty: float, weight for KL divergence term
    """

    def __init__(self, kl_penalty: float = 1e-5) -> None:
        """Initializes an instance of the ELBOLoss class.

        Args:
        kl_penalty: float, weight for KL divergence term
        """
        super().__init__()
        self.kl_penalty = kl_penalty

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, kl: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ELBO loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)
            kl: torch.Tensor, KL divergence of the model
        Returns:
            loss: torch.Tensor, mean loss value
        """
        loss = F.cross_entropy(inputs, targets) + self.kl_penalty * kl
        return loss
