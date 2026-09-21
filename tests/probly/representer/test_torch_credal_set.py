"""Tests for ``probly.representer.torch_credal_set``."""

from __future__ import annotations

import subprocess
import sys
from typing import Literal

import pytest

torch = pytest.importorskip("torch")


class TestRepresenterCredalSetTorch:
    """Direct call to the torch dispatch for representative-sample computation."""

    def test_torch_handler_alpha_zero_returns_input(self) -> None:
        """Calling the torch handler directly with alpha=0 short-circuits."""
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.torch_credal_set import torch_compute_representative_sample  # noqa: PLC0415

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
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.torch_credal_set import torch_compute_representative_sample  # noqa: PLC0415

        dist = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[[0.5, 0.5]]]))
        sample = TorchSample(tensor=dist, sample_dim=0)
        with pytest.raises(NotImplementedError, match="not implemented"):
            torch_compute_representative_sample(sample, alpha=0.5, distance="manhattan")

    def test_torch_handler_filters_to_top_k(self) -> None:
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.torch_credal_set import torch_compute_representative_sample  # noqa: PLC0415

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
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.torch_credal_set import torch_compute_representative_sample  # noqa: PLC0415

        dist = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[[0.4, 0.4, 0.2]], [[0.6, 0.3, 0.1]]]))
        sample = TorchSample(tensor=dist, sample_dim=0)
        result = torch_compute_representative_sample(sample, alpha=1.0, distance="euclidean")
        # k = max(int(2 * (1 - 1)), 1) = 1
        assert result.sample_size == 1


@pytest.mark.parametrize("sample_axis", [0, 1])
@pytest.mark.parametrize("typed_sample", [False, True])
def test_lazy_registration_imports_renamed_module(sample_axis: int, typed_sample: bool) -> None:
    code = f"""
import sys
import torch
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.sample.torch import TorchSample
from probly.representer.credal_set import compute_representative_sample

module = 'probly.representer.torch_credal_set'
assert module not in sys.modules
distribution = TorchProbabilityCategoricalDistribution(torch.tensor([[[0.4, 0.6]], [[0.5, 0.5]]]))
sample_type = TorchCategoricalDistributionSample if {typed_sample!r} else TorchSample
sample = sample_type(distribution, sample_dim=0).move_sample_axis({sample_axis})
assert compute_representative_sample(sample, alpha=0.0, distance='euclidean') is sample
assert module in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("sample_axis", [0, 1, -1, "auto"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("typed_sample", [False, True])
def test_public_filtering_is_independent_of_sample_axis(
    sample_axis: int | Literal["auto"], alpha: float, typed_sample: bool
) -> None:
    from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
        TorchCategoricalDistributionSample,
        TorchProbabilityCategoricalDistribution,
    )
    from probly.representation.sample import create_sample  # noqa: PLC0415
    from probly.representer.credal_set import compute_representative_sample  # noqa: PLC0415

    probabilities = torch.tensor(
        [
            [[0.4, 0.6], [0.9, 0.1]],
            [[0.55, 0.45], [0.3, 0.7]],
            [[0.1, 0.9], [0.4, 0.6]],
            [[0.8, 0.2], [0.1, 0.9]],
        ]
    )
    predictions = [TorchProbabilityCategoricalDistribution(probs) for probs in probabilities]
    sample = create_sample(predictions, sample_axis=sample_axis)
    if typed_sample:
        sample = TorchCategoricalDistributionSample(sample.tensor, sample_dim=sample.sample_dim)

    filtered = compute_representative_sample(sample, alpha=alpha, distance="euclidean")

    assert filtered.sample_dim == sample.sample_dim
    if alpha == 0.0:
        assert filtered is sample
        expected = probabilities
    else:
        # Select the nearest members independently for each batch item.
        expected = torch.tensor([[[0.4, 0.6], [0.4, 0.6]], [[0.55, 0.45], [0.3, 0.7]]])
        if alpha == 1.0:
            expected = expected[:1]
    assert filtered.sample_size == len(expected)
    torch.testing.assert_close(filtered.tensor.probabilities.movedim(filtered.sample_dim, 0), expected)


def test_public_filtering_rejects_non_categorical_samples() -> None:
    from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
    from probly.representer.credal_set import compute_representative_sample  # noqa: PLC0415

    sample = TorchSample(torch.ones(2, 3), sample_dim=0)
    with pytest.raises(NotImplementedError, match="No representative-sample computation registered"):
        compute_representative_sample(sample, alpha=0.0, distance="euclidean")


def test_credal_ensembling_representer_preserves_ensemble_predictions() -> None:
    from probly.method.credal_ensembling import credal_ensembling  # noqa: PLC0415
    from probly.representation.credal_set.torch import TorchConvexCredalSet  # noqa: PLC0415
    from probly.representer import representer  # noqa: PLC0415

    ensemble = credal_ensembling(torch.nn.Linear(2, 3), num_members=4, predictor_type="logit_classifier")
    inputs = torch.tensor([[0.0, 1.0], [1.0, -1.0]])
    with torch.no_grad():
        credal_set = representer(ensemble).predict(inputs)
        expected = torch.stack([member(inputs).softmax(-1) for member in ensemble], dim=1)

    assert isinstance(credal_set, TorchConvexCredalSet)
    torch.testing.assert_close(credal_set.tensor.probabilities, expected)
