"""Tests for torch-backed categorical credal sets."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation.credal_set._common import (
    create_distance_based_credal_set_from_center_and_radius,
)
from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.sample.torch import TorchSample


def _torch_modules():
    """Return torch module or skip the calling test."""
    pytest.importorskip("torch")
    import torch as _torch  # noqa: PLC0415

    return _torch


def test_torch_convex_credal_set_from_distribution_sample() -> None:
    probs = torch.tensor(
        [
            [[0.1, 0.9], [0.4, 0.6]],
            [[0.2, 0.8], [0.5, 0.5]],
            [[0.15, 0.85], [0.45, 0.55]],
        ],
        dtype=torch.float64,
    )
    sample = TorchSample(
        tensor=TorchProbabilityCategoricalDistribution(probs),
        sample_dim=0,
    )

    cset = TorchConvexCredalSet.from_torch_sample(sample)

    assert isinstance(cset.tensor, TorchCategoricalDistribution)
    assert tuple(cset.tensor.probabilities.shape) == (2, 3, 2)


def test_torch_convex_credal_set_barycenter_averages_normalized_probabilities() -> None:
    vertices = torch.tensor([[1.0, 1.0], [9.0, 1.0]], dtype=torch.float64)
    cset = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))

    barycenter = cset.barycenter

    assert isinstance(barycenter, TorchCategoricalDistribution)
    assert torch.allclose(barycenter.probabilities, torch.tensor([0.7, 0.3], dtype=torch.float64))


def test_torch_probability_intervals_numpy_and_shape_ops() -> None:
    probs = torch.tensor(
        [
            [[0.2, 0.8], [0.6, 0.4]],
            [[0.1, 0.9], [0.5, 0.5]],
        ],
        dtype=torch.float64,
    )
    sample = TorchSample(
        tensor=TorchProbabilityCategoricalDistribution(probs),
        sample_dim=0,
    )

    cset = TorchProbabilityIntervalsCredalSet.from_torch_sample(sample)
    arr = np.asarray(cset)

    assert arr.shape == (2, 2, 2)

    expanded = torch.unsqueeze(cset, dim=0)
    assert isinstance(expanded, TorchProbabilityIntervalsCredalSet)
    assert tuple(expanded.lower_bounds.shape) == (1, 2, 2)
    assert tuple(expanded.upper_bounds.shape) == (1, 2, 2)


def test_distance_credal_set_from_categorical_distribution() -> None:
    """Factory should accept TorchCategoricalDistribution directly."""
    probs = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]], dtype=torch.float64)
    center = TorchProbabilityCategoricalDistribution(probs)
    radius = torch.tensor(0.1, dtype=torch.float64)

    result = create_distance_based_credal_set_from_center_and_radius(center, radius)

    assert isinstance(result, TorchDistanceBasedCredalSet)
    assert result.nominal is center  # should reuse, not re-wrap
    # Scalar radius is broadcast to one entry per sample so torch.cat across
    # batches works under the protected-axes contract.
    expected_radius = radius.expand(probs.shape[0])
    assert torch.equal(result.radius, expected_radius)


class TestTorchConvexCredalSet:
    """Convex credal sets backed by torch tensors."""

    def test_from_torch_sample(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchConvexCredalSet  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        # 3 samples, 1 batch, 2 classes.
        probs = torch.tensor([[[0.5, 0.5]], [[0.3, 0.7]], [[0.6, 0.4]]])
        sample = TorchSample(
            tensor=TorchProbabilityCategoricalDistribution(probs),
            sample_dim=0,
        )
        cred = TorchConvexCredalSet.from_torch_sample(sample)
        # The vertices should have num_vertices on a class-axis-leading layout.
        assert cred.tensor.tensor.shape[-1] == 2

    def test_lower_upper(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchConvexCredalSet  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        probs = torch.tensor([[[0.5, 0.5], [0.3, 0.7]]])
        cred = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(probs))
        torch.testing.assert_close(cred.lower(), torch.tensor([[0.3, 0.5]]))
        torch.testing.assert_close(cred.upper(), torch.tensor([[0.5, 0.7]]))

    def test_num_classes(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchConvexCredalSet  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        probs = torch.tensor([[[0.5, 0.5], [0.3, 0.7]]])
        cred = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(probs))
        assert cred.num_classes == 2


class TestTorchDistanceBasedCredalSet:
    """Distance-based credal sets backed by torch tensors."""

    def test_from_torch_sample(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchDistanceBasedCredalSet  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        probs = torch.tensor([[[0.5, 0.5]], [[0.3, 0.7]]])
        sample = TorchSample(
            tensor=TorchProbabilityCategoricalDistribution(probs),
            sample_dim=0,
        )
        cred = TorchDistanceBasedCredalSet.from_torch_sample(sample)
        # Mean of [0.5, 0.5] and [0.3, 0.7] is [0.4, 0.6]. The TV-distance from
        # the mean to either is 0.1.
        torch.testing.assert_close(cred.nominal.tensor.squeeze(), torch.tensor([0.4, 0.6]), atol=1e-5, rtol=1e-5)
        # Radius should equal max TV distance.
        assert cred.radius.shape == (1,)

    def test_lower_upper(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchDistanceBasedCredalSet  # noqa: PLC0415

        cred = TorchDistanceBasedCredalSet(
            nominal=torch.tensor([[0.4, 0.6]]),
            radius=torch.tensor([0.1]),
        )
        torch.testing.assert_close(cred.lower(), torch.tensor([[0.3, 0.5]]))
        torch.testing.assert_close(cred.upper(), torch.tensor([[0.5, 0.7]]))

    def test_lower_clipped(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchDistanceBasedCredalSet  # noqa: PLC0415

        cred = TorchDistanceBasedCredalSet(
            nominal=torch.tensor([[0.05, 0.95]]),
            radius=torch.tensor([0.5]),
        )
        torch.testing.assert_close(cred.lower(), torch.tensor([[0.0, 0.45]]))

    def test_upper_clipped(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchDistanceBasedCredalSet  # noqa: PLC0415

        cred = TorchDistanceBasedCredalSet(
            nominal=torch.tensor([[0.95, 0.05]]),
            radius=torch.tensor([0.5]),
        )
        torch.testing.assert_close(cred.upper(), torch.tensor([[1.0, 0.55]]))


class TestTorchProbabilityIntervalsCredalSet:
    """Probability-interval credal sets backed by torch tensors."""

    def test_width(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet  # noqa: PLC0415

        cred = TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.tensor([[0.1, 0.2]]),
            upper_bounds=torch.tensor([[0.5, 0.6]]),
        )
        torch.testing.assert_close(cred.width(), torch.tensor([[0.4, 0.4]]))

    def test_contains(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet  # noqa: PLC0415

        cred = TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.tensor([[0.1, 0.2]]),
            upper_bounds=torch.tensor([[0.5, 0.6]]),
        )
        # Inside intervals -> True.
        assert bool(cred.contains(torch.tensor([[0.3, 0.4]])))
        # Outside intervals -> False.
        assert not bool(cred.contains(torch.tensor([[0.7, 0.5]])))

    def test_numpy_method(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet  # noqa: PLC0415

        cred = TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.tensor([[0.1, 0.2]]),
            upper_bounds=torch.tensor([[0.5, 0.6]]),
        )
        arr = cred.numpy()
        # Shape should be (1, 2, 2): batch x [lower/upper] x classes.
        assert arr.shape == (1, 2, 2)

    def test_numpy_method_force_copy(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet  # noqa: PLC0415

        cred = TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.tensor([[0.1, 0.2]]),
            upper_bounds=torch.tensor([[0.5, 0.6]]),
        )
        arr = cred.numpy(force=True)
        assert arr.shape == (1, 2, 2)

    def test_num_classes(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet  # noqa: PLC0415

        cred = TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.tensor([[0.1, 0.2, 0.3]]),
            upper_bounds=torch.tensor([[0.4, 0.5, 0.6]]),
        )
        assert cred.num_classes == 3


class TestEnsureTorchCategoricalDistribution:
    """The internal _ensure_torch_categorical_distribution coerces inputs."""

    def test_passthrough(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import _ensure_torch_categorical_distribution  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        d = TorchProbabilityCategoricalDistribution(torch.tensor([[0.5, 0.5]]))
        assert _ensure_torch_categorical_distribution(d) is d

    def test_wraps_tensor(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import _ensure_torch_categorical_distribution  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: PLC0415

        d = _ensure_torch_categorical_distribution(torch.tensor([[0.5, 0.5]]))
        assert isinstance(d, TorchCategoricalDistribution)


class TestSampleProbabilities:
    """The internal _sample_probabilities helper rejects non-categorical samples."""

    def test_non_categorical_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set.torch import _sample_probabilities  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        # Plain tensor sample, not a categorical distribution.
        sample = TorchSample(tensor=torch.tensor([[0.5, 0.5], [0.3, 0.7]]), sample_dim=0)
        with pytest.raises(TypeError, match="TorchCategoricalDistribution"):
            _sample_probabilities(sample)


class TestCreateFromBounds:
    """`create_probability_intervals_from_lower_upper_array` and `_from_bounds`."""

    def test_create_intervals_from_packed_bounds(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set._common import (  # noqa: PLC0415
            create_probability_intervals_from_lower_upper_array,
        )

        # Force registration of torch handler.
        import probly.representation.credal_set.torch as _torch  # noqa: F401, PLC0415
        from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet  # noqa: PLC0415

        # Shape (..., 2 * num_classes) packed format.
        packed = torch.tensor([[0.1, 0.2, 0.5, 0.6]])
        cred = create_probability_intervals_from_lower_upper_array(packed)
        assert isinstance(cred, TorchProbabilityIntervalsCredalSet)
        # First half is lower, second half is upper.
        torch.testing.assert_close(cred.lower_bounds, torch.tensor([[0.1, 0.2]]))
        torch.testing.assert_close(cred.upper_bounds, torch.tensor([[0.5, 0.6]]))

    def test_create_intervals_from_bounds_separate_args(self) -> None:
        torch = _torch_modules()
        from probly.representation.credal_set._common import create_probability_intervals_from_bounds  # noqa: PLC0415

        # Force registration.
        import probly.representation.credal_set.torch as _torch  # noqa: F401, PLC0415
        from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet  # noqa: PLC0415

        probs = torch.tensor([[0.4, 0.5]])
        lower = torch.tensor([[0.1, 0.1]])
        upper = torch.tensor([[0.1, 0.1]])
        cred = create_probability_intervals_from_bounds(probs, lower, upper)
        assert isinstance(cred, TorchProbabilityIntervalsCredalSet)
