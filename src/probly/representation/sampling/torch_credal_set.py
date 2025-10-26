"""Torch credal set implementation."""

from __future__ import annotations

import torch

from probly.representation.sampling.credal_set import CredalSet, create_credal_set


@create_credal_set.register(torch.Tensor)
class TorchTensorCredalSet(CredalSet[torch.Tensor]):
    """A credal set implementation for torch tensors."""

    def __init__(self, samples: list[torch.Tensor]) -> None:
        """Initialize the torch tensor credal set."""
        self.tensor = torch.stack(samples).permute(1, 0, 2)  # we use the convention [instances, samples, classes]

    def lower(self) -> torch.Tensor:
        """Compute the lower envelope of the credal set."""
        return self.tensor.min(dim=1).values
