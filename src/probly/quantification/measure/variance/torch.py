"""Torch implementations of regression measures."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchSample

from ._common import LogBase, expected_conditional_variance


@expected_conditional_variance.register(TorchSample)
def torch_sample_expected_conditional_variance(
    sample: TorchSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the expected conditional variance of a raw tensor sample."""
    return torch.zeros_like(torch.mean(sample.tensor, dim=sample.sample_axis))
