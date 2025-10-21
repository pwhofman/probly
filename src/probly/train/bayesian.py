"""Functions for training Bayesian neural networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probly.predictor import Predictor

import torch

from probly.layers.torch import BayesConv2d, BayesLinear  # noqa: TC001, required by traverser
from probly.traverse_nn import nn_compose
from pytraverse import GlobalVariable, State, TraverserResult, singledispatch_traverser, traverse_with_state

KL_DIVERGENCE = GlobalVariable[torch.Tensor | float]("KL_DIVERGENCE", default=0.0)


@singledispatch_traverser[object]
def kl_divergence_traverser(
    obj: BayesLinear | BayesConv2d,
    state: State,
) -> TraverserResult[BayesLinear | BayesConv2d]:
    """Traverser to compute the KL divergence of a Bayesian layer."""
    state[KL_DIVERGENCE] += obj.kl_divergence
    return obj, state


def collect_kl_divergence(model: Predictor) -> torch.Tensor | float:
    """Collect the KL divergence of the Bayesian model by summing the KL divergence of each Bayesian layer."""
    _, state = traverse_with_state(model, nn_compose(kl_divergence_traverser))
    return state[KL_DIVERGENCE]
