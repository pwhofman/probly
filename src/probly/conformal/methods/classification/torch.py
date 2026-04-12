from __future__ import annotations

from torch import nn
import torch

from probly.layers.torch import ConformalClassificationHead

from ._common import conformal_generator, to_probabilities


@conformal_generator.register(nn.Module)
def _(model: nn.Module) -> nn.Module:
    """Conformalise a PyTorch model."""
    return nn.Sequential(
        model,
        ConformalClassificationHead(),
    )


@to_probabilities.register(torch.Tensor)
def _(pred: torch.Tensor) -> torch.Tensor:
    """Obtain probabilities from a PyTorch model."""
    if pred.ndim != 2:
        msg = f"Probability extraction expects a 2D array, got {pred.ndim}D array instead."
        raise ValueError(msg)
    if torch.allclose(pred.sum(dim=-1), torch.ones(pred.shape[0], device=pred.device)):
        # If the predictions already sum to 1, we assume they are probabilities
        return pred
    probs = torch.nn.functional.softmax(pred, dim=-1)
    return probs
