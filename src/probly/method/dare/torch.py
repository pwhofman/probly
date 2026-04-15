"""Torch dare implementation."""

from __future__ import annotations

import torch
from torch import nn

from probly.method.ensemble import EnsemblePredictor, ensemble
from probly.predictor._common import predict, predict_raw

from ._common import DarePredictor, dare_generator


class DareModule(nn.Module):
    """A wrapper for DARE ensemble members.

    Attributes:
        members: The list of ensemble members.
        lambda_reg: The anti-regularization coefficient.
        threshold: The loss threshold to trigger anti-regularization.
    """

    def __init__(
        self,
        ensemble_members: EnsemblePredictor,
        lambda_reg: float = 0.01,
        threshold: float = 0.1,
    ) -> None:
        """Initialize the DareModule."""
        super().__init__()
        self.members = ensemble_members
        self.lambda_reg = lambda_reg
        self.threshold = threshold

    def forward[**In](self, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
        """Forward pass for all members.

        Returns:
            A stacked tensor of predictions from all members.
        """
        tensors = [predict(member, *args, **kwargs) for member in self.members]
        return torch.stack(tensors, dim=0)

    def __len__(self) -> int:
        """Returns the length of the ensemble."""
        return len(self.members)

    def __getitem__(self, idx: int) -> nn.Module:
        """Returns a member from an index."""
        return self.members[idx]

    def __iter__(self) -> nn.Module:
        """Returns an iterator over the members."""
        return iter(self.members)


DarePredictor.register(DareModule)


@dare_generator.register(nn.Module)
def torch_dare_wrapper(
    obj: nn.Module, num_members: int, lambda_reg: float = 0.01, threshold: float = 0.1
) -> DareModule:
    """Creates a DARE ensemble.

    Args:
        obj: The torch base module.
        num_members: The number of members in the dare.
        lambda_reg: The anti-regularization coefficient.
        threshold: The loss threshold for switching anti-regularization on.

    Returns:
        The DareModule containing the ensemble.
    """
    ensemble_members = ensemble(obj, num_members=num_members, reset_params=False)
    return DareModule(ensemble_members, lambda_reg=lambda_reg, threshold=threshold)


@predict_raw.register(DareModule)
def predict_dare_module[**In](predictor: DareModule, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
    """Predict for a dare module."""
    return predictor(*args, **kwargs)
