"""Collection of torch calibration training functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def label_relaxation_loss(inputs: torch.Tensor, targets: torch.Tensor, *, alpha: float = 0.1) -> torch.Tensor:
    """Label Relaxation Loss from :cite:`lienenFromLabel2021`.

    This loss is used to improve the calibration of a neural network. It works by minimizing
    the Kullback-Leibler divergence between the predicted probabilities and the target distribution in the credal set
    defined by the alpha parameter. The target distribution is the distribution in the credal set that minimizes the
    Kullback-Leibler divergence from the predicted probabilities. If the predicted probability distribution
    is in the credal set, the loss is zero.

    Args:
        inputs: Logits of size (n_instances, n_classes).
        targets: Class labels of size (n_instances,).
        alpha: The parameter that controls the amount of label relaxation. Increasing alpha, increases the size
            of the credal set and thus the amount of label relaxation.

    Returns:
        The mean loss value.
    """
    inputs_probs = F.softmax(inputs, dim=1)

    with torch.no_grad():
        inv_one_hot = 1 - F.one_hot(targets, inputs.shape[1])
        targets_real = alpha * inputs_probs / torch.sum(inv_one_hot * inputs_probs, dim=1, keepdim=True)
        targets_real[torch.arange(targets.shape[0]), targets] = 1 - alpha

    kl_div = torch.sum(F.kl_div(inputs_probs.log(), targets_real, log_target=False, reduction="none"), dim=1)
    loss = torch.where(torch.sum(inv_one_hot * inputs_probs, dim=1) <= alpha, 0, kl_div)
    return loss.mean()


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, *, alpha: float = 1, gamma: float = 2) -> torch.Tensor:
    """Focal Loss based on :cite:`linFocalLoss2017`.

    Args:
        inputs: Logits of size (n_instances, n_classes).
        targets: Class labels of size (n_instances,).
        alpha: Control importance of minority class.
        gamma: Control loss for hard instances.

    Returns:
        The mean loss value.
    """
    targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])
    prob = F.softmax(inputs, dim=-1)
    p_t = torch.sum(prob * targets_one_hot, dim=-1)

    log_prob = torch.log(prob)
    loss = -alpha * (1 - p_t) ** gamma * torch.sum(log_prob * targets_one_hot, dim=-1)

    return torch.mean(loss)
