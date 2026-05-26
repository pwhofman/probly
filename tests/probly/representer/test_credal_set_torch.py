"""Tests for ``probly.representer.credal_set_torch``."""

from __future__ import annotations

import pytest


def _torch():
    return pytest.importorskip("torch")


class TestRepresenterCredalSetTorch:
    """Direct call to the torch dispatch for representative-sample computation."""

    def test_torch_handler_alpha_zero_returns_input(self) -> None:
        """Calling the torch handler directly with alpha=0 short-circuits."""
        torch = _torch()
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.credal_set_torch import torch_compute_representative_sample  # noqa: PLC0415

        dist = TorchProbabilityCategoricalDistribution(
            tensor=torch.tensor(
                [
                    [[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]],
                    [[0.4, 0.4, 0.2], [0.2, 0.5, 0.3]],
                ]
            )
        )
        sample = TorchSample(tensor=dist, sample_dim=0)
        result = torch_compute_representative_sample(sample, alpha=0.0, distance="euclidean")
        assert result is sample

    def test_torch_handler_unsupported_distance_raises(self) -> None:
        torch = _torch()
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.credal_set_torch import torch_compute_representative_sample  # noqa: PLC0415

        dist = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[[0.5, 0.5]]]))
        sample = TorchSample(tensor=dist, sample_dim=0)
        with pytest.raises(NotImplementedError, match="not implemented"):
            torch_compute_representative_sample(sample, alpha=0.5, distance="manhattan")

    def test_torch_handler_filters_to_top_k(self) -> None:
        torch = _torch()
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.credal_set_torch import torch_compute_representative_sample  # noqa: PLC0415

        # 4 samples, 1 batch, 3 classes.
        dist = TorchProbabilityCategoricalDistribution(
            tensor=torch.tensor(
                [
                    [[0.4, 0.4, 0.2]],
                    [[0.5, 0.3, 0.2]],
                    [[0.1, 0.1, 0.8]],
                    [[0.8, 0.1, 0.1]],
                ]
            )
        )
        sample = TorchSample(tensor=dist, sample_dim=0)
        result = torch_compute_representative_sample(sample, alpha=0.5, distance="euclidean")
        # alpha=0.5 -> keep half (k=2).
        assert result.sample_size == 2

    def test_torch_handler_alpha_keeps_at_least_one(self) -> None:
        """Even alpha=1.0 keeps at least one sample (k >= 1)."""
        torch = _torch()
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.credal_set_torch import torch_compute_representative_sample  # noqa: PLC0415

        dist = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[[0.4, 0.4, 0.2]], [[0.6, 0.3, 0.1]]]))
        sample = TorchSample(tensor=dist, sample_dim=0)
        result = torch_compute_representative_sample(sample, alpha=1.0, distance="euclidean")
        # k = max(int(2 * (1 - 1)), 1) = 1
        assert result.sample_size == 1
