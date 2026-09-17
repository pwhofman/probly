"""Collection of torch Bayesian training functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def elbo_loss(
    inputs: torch.Tensor, targets: torch.Tensor, kl: torch.Tensor, *, kl_penalty: float = 1e-5
) -> torch.Tensor:
    """Evidence lower bound loss based on :cite:`blundellWeightUncertainty2015`.

    Args:
        inputs: Logits of size (n_instances, n_classes).
        targets: Class labels of size (n_instances,).
        kl: KL divergence of the model.
        kl_penalty: Weight for KL divergence term.

    Returns:
        The mean loss value.
    """
    return F.cross_entropy(inputs, targets) + kl_penalty * kl
