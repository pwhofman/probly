"""Collection of torch Bayesian training functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probly.predictor import Predictor

import torch
from torch import nn
import torch.nn.functional as F

from probly.layers.torch import BayesConv2d, BayesLinear  # noqa: TC001, required by traverser
from probly.traverse_nn import nn_compose
from pytraverse import GlobalVariable, State, TraverserResult, singledispatch_traverser, traverse_with_state

KL_DIVERGENCE = GlobalVariable[torch.Tensor]("KL_DIVERGENCE", default=0.0)


@singledispatch_traverser[object]
def kl_divergence_traverser(
    obj: BayesLinear | BayesConv2d,
    state: State,
) -> TraverserResult[BayesLinear | BayesConv2d]:
    """Traverser to compute the KL divergence of a Bayesian layer based on :cite:`galDropoutBayesian2016`."""
    state[KL_DIVERGENCE] += obj.kl_divergence
    return obj, state


def collect_kl_divergence(model: Predictor) -> torch.Tensor:
    """Collect the KL divergence of the Bayesian model by summing the KL divergence of each Bayesian layer."""
    _, state = traverse_with_state(model, nn_compose(kl_divergence_traverser))
    return state[KL_DIVERGENCE]


class ELBOLoss(nn.Module):
    """Evidence lower bound loss based on :cite:`blundellWeightUncertainty2015`.

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
