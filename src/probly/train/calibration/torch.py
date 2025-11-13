"""Collection of torch calibration training functions."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ExpectedCalibrationError(nn.Module):
    """Expected Calibration Error (ECE) :cite:`guoOnCalibration2017`.

    Attributes:
        num_bins: int, number of bins to use for calibration
        self.bins: torch.Tensor, the actual bins for calibration
    """

    def __init__(self, num_bins: int = 10) -> None:
        """Initializes an instance of the ExpectedCalibrationError class.

        Args:
            num_bins: int, number of bins to use for calibration
        """
        super().__init__()
        self.num_bins = num_bins
        self.bins = torch.linspace(0, 1, num_bins + 1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the expected calibration error.

        Assumes that inputs are probability distributions over classes.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes).
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value
        """
        confs, preds = torch.max(inputs, dim=1)
        bin_indices = torch.bucketize(confs, self.bins.to(inputs.device), right=True) - 1
        num_instances = inputs.shape[0]
        loss: torch.Tensor = torch.tensor(0, dtype=torch.float32, device=inputs.device)
        for i in range(self.num_bins):
            _bin = torch.where(bin_indices == i)[0]
            # check if bin is empty
            if _bin.shape[0] == 0:
                continue
            acc_bin = torch.mean((preds[_bin] == targets[_bin]).float())
            conf_bin = torch.mean(confs[_bin])
            weight = _bin.shape[0] / num_instances
            loss += weight * torch.abs(acc_bin - conf_bin)
        return loss


class LabelRelaxationLoss(nn.Module):
    """Label Relaxation Loss from :cite:`lienenFromLabel2021`.

    This loss is used to improve the calibration of a neural network. It works by minimizing
    the Kullback-Leibler divergence between the predicted probabilities and the target distribution in the credal set
    defined by the alpha parameter. The target distribution is the distribution in the credal set that minimizes the
    Kullback-Leibler divergence from the predicted probabilities. If the predicted probability distribution
    is in the credal set, the loss is zero.

    Attributes:
        alpha: float, the parameter that controls the amount of label relaxation. Increasing alpha, increases the size
            of the credal set and thus the amount of label relaxation.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        """Initializes an instance of the LabelRelaxationLoss class.

        Args:
            alpha: float, the parameter that controls the amount of label relaxation.
                Increasing alpha, increases the size of the credal set and thus the amount of label relaxation.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the label relaxation loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value
        """
        inputs_probs = F.softmax(inputs, dim=1)

        with torch.no_grad():
            inv_one_hot = 1 - F.one_hot(targets, inputs.shape[1])
            targets_real = self.alpha * inputs_probs / torch.sum(inv_one_hot * inputs_probs, dim=1, keepdim=True)
            targets_real[torch.arange(targets.shape[0]), targets] = 1 - self.alpha

        kl_div = torch.sum(F.kl_div(inputs_probs.log(), targets_real, log_target=False, reduction="none"), dim=1)
        loss = torch.where(torch.sum(inv_one_hot * inputs_probs, dim=1) <= self.alpha, 0, kl_div)
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss based on :cite:`linFocalLoss2017`.

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
