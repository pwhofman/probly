import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialLogLoss(nn.Module):
    """
    Evidential Log Loss based on https://arxiv.org/pdf/1806.01768.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the evidential log loss.
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
    """
    Evidential Cross Entropy Loss based on https://arxiv.org/pdf/1806.01768.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the evidential cross entropy loss.
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
    """
    Evidential Mean Square Error Loss based on https://arxiv.org/pdf/1806.01768.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the evidential mean squared error loss.
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
    """
    Evidential KL Divergence Loss based on https://arxiv.org/pdf/1806.01768.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the evidential KL divergence loss.
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
        K = torch.full((inputs.shape[0],), inputs.shape[1])
        first = (torch.lgamma(strengths_tilde) -
                 torch.lgamma(K) -
                 torch.sum(torch.lgamma(alphas_tilde), dim=1)
                 )
        second = torch.sum((alphas_tilde - 1) * (torch.digamma(alphas_tilde) - torch.digamma(strengths_tilde)[:, None]), dim=1)
        loss = torch.mean(first + second)
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss based on https://arxiv.org/pdf/1708.02002
    Args:
        alpha: float, control importance of minority class
        gamma: float, control loss for hard instances
        reduction: str 'mean' or 'sum' or None, method of reduction towards single loss value
    """

    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the focal loss.
        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)
        Returns:
            loss: torch.Tensor size (1,) or (n_instances,), loss value
        """
        log_prob = F.log_softmax(inputs, dim=-1)  # TODO check other way around and test function
        prob = torch.exp(log_prob)  # calculate model confidence

        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])
        p_t = torch.sum(prob * targets_one_hot, dim=-1)

        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.sum(log_prob * targets_one_hot, dim=-1)

        # Apply reduction (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
