"""Tests for sample-based torch uncertainty measures."""

from __future__ import annotations

import pytest


def _torch_modules():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    return torch


class TestSampleMeasureTorch:
    """Sample-based torch uncertainty measures."""

    def test_mean_squared_distance_to_scaled_one_hot(self) -> None:
        torch = _torch_modules()
        from probly.quantification.measure.sample import (  # noqa: PLC0415
            mean_squared_distance_to_scaled_one_hot,
        )
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchCategoricalDistributionSample,
            TorchProbabilityCategoricalDistribution,
        )

        # 2 members of a 3-class categorical for batch=1.
        probs = torch.tensor([[[0.6, 0.3, 0.1]], [[0.1, 0.6, 0.3]]])
        dist = TorchProbabilityCategoricalDistribution(tensor=probs)
        sample = TorchCategoricalDistributionSample(tensor=dist, sample_dim=0)
        result = mean_squared_distance_to_scaled_one_hot(sample, scale=1.0)
        # Result is per-batch.
        assert result.shape == (1,)
        assert torch.isfinite(result).all()

    def test_total_logit_sample_variance_torch(self) -> None:
        torch = _torch_modules()
        from probly.quantification.measure.sample import total_logit_sample_variance  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchCategoricalDistributionSample,
            TorchProbabilityCategoricalDistribution,
        )

        probs = torch.tensor([[[0.6, 0.3, 0.1]], [[0.1, 0.6, 0.3]]])
        dist = TorchProbabilityCategoricalDistribution(tensor=probs)
        sample = TorchCategoricalDistributionSample(tensor=dist, sample_dim=0)
        result = total_logit_sample_variance(sample)
        assert result.shape == (1,)
        assert torch.isfinite(result).all()
