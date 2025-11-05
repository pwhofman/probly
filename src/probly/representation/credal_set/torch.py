"""Torch credal set implementation."""

from __future__ import annotations

import torch

from probly.representation.credal_set.credal_set import CredalSet, credal_set_from_sample
from probly.representation.sampling.torch_sample import TorchTensorSample


@credal_set_from_sample.register(TorchTensorSample)
class TorchTensorCredalSet(CredalSet[torch.Tensor]):
    """A credal set implementation for torch tensors."""

    def __init__(self, sample: TorchTensorSample) -> None:
        """Initialize the torch tensor credal set."""
        self.tensor = sample.tensor

    def lower(self) -> torch.Tensor:
        """Compute the lower envelope of the credal set."""
        return self.tensor.min(dim=1).values

    def upper(self) -> torch.Tensor:
        """Compute the upper envelope of the credal set."""
        return self.tensor.max(dim=1).values
