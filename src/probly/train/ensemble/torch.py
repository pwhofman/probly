"""Collection of torch ensemble training functions."""

from __future__ import annotations

import torch
from torch import nn


class RegularizerDare(nn.Module):
    def __init__(self, lambda_reg: float = 0.01) -> None:
        """Ensemble Regularizer based on :cite:`<TODO:DARE PAPER>`"""
        super().__init__()
        self.lambda_reg: float = lambda_reg

    def forward(self, model: nn.Module) -> torch.Tensor:
        total_loss: torch.Tensor = torch.tensor(0.0, requires_grad=True)
        for param in model.parameters():
            total_loss += param.square().log2().sum()
        return total_loss * self.lambda_reg
