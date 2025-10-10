"""Torch sample implementation."""

from __future__ import annotations

import torch

from .sample import Sample, create_sample


@create_sample.register(torch.Tensor)
class TorchTensorSample(Sample[torch.Tensor]):
    """A sample implementation for torch tensors."""

    def __init__(self, samples: list[torch.Tensor]) -> None:
        """Initialize the torch tensor sample."""
        self.tensor = torch.stack(samples)

    def mean(self) -> torch.Tensor:
        """Compute the mean of the sample."""
        return self.tensor.mean(dim=0)

    def std(self, ddof: int = 1) -> torch.Tensor:
        """Compute the standard deviation of the sample."""
        return self.tensor.std(dim=0, correction=ddof)

    def var(self, ddof: int = 1) -> torch.Tensor:
        """Compute the variance of the sample."""
        return self.tensor.var(dim=0, correction=ddof)
