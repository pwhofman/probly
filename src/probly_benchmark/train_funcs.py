"""Training functionality for probly benchmark methods."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any

import torch
from torch import nn, optim

from probly.method.bayesian import BayesianPredictor

if TYPE_CHECKING:
    from probly.predictor import Predictor


@singledispatch
def train_epoch(
    model: Predictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    **kwargs: Any,  # noqa: ANN401
) -> torch.Tensor | float:
    """Train for one epoch."""
    msg = f"No training function for {type(model)}"
    raise NotImplementedError(msg)


@train_epoch.register
def _(
    model: BayesianPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> torch.Tensor | float:
    """Train a Bayesian predictor for one epoch."""
    optimizer.zero_grad()
    outputs = model(inputs)  # ty: ignore
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()
