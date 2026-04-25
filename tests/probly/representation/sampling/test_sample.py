"""Test sample dispatching logic."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("jax")
from jax import numpy as jnp
import numpy as np
import torch

from probly.representation.sample import ArraySample, ListSample, create_sample
from probly.representation.sample.jax import JaxArraySample
from probly.representation.sample.torch import TorchSample


class TestSampleDispatching:
    def test_create_array_sample_numpy(self) -> None:
        x = np.arange(12).reshape((3, 4))
        sample = create_sample(x)
        assert isinstance(sample, ArraySample)
        assert sample.shape == (4, 3)
        assert sample.sample_axis == 1

    def test_create_array_sample_jax(self) -> None:
        x = jnp.arange(12).reshape((3, 4))
        sample = create_sample(x, sample_axis=1)
        assert isinstance(sample, JaxArraySample)
        assert sample.shape == (4, 3)
        assert sample.sample_axis == 1

    def test_create_array_sample_torch(self) -> None:
        x = torch.arange(12).reshape((3, 4))
        sample = create_sample(x, sample_axis=0)
        assert isinstance(sample, TorchSample)
        assert sample.shape == (3, 4)
        assert sample.sample_dim == 0
        assert x is sample.tensor

    def test_create_array_sample_numpy_preserves_weights(self) -> None:
        x = np.arange(12).reshape((3, 4))
        weights = np.array([0.1, 0.2, 0.3])

        sample = create_sample(x, sample_axis=0, weights=weights)

        assert isinstance(sample, ArraySample)
        assert np.array_equal(sample.weights, weights)

    def test_create_array_sample_jax_preserves_weights(self) -> None:
        x = jnp.arange(12).reshape((3, 4))
        weights = jnp.array([0.1, 0.2, 0.3])

        sample = create_sample(x, sample_axis=0, weights=weights)

        assert isinstance(sample, JaxArraySample)
        assert np.array_equal(np.asarray(sample.weights), np.asarray(weights))

    def test_create_array_sample_torch_preserves_weights(self) -> None:
        x = torch.arange(12).reshape((3, 4))
        weights = torch.tensor([0.1, 0.2, 0.3])

        sample = create_sample(x, sample_axis=0, weights=weights)

        assert isinstance(sample, TorchSample)
        assert torch.equal(sample.weights, weights)


class TestListSampleWeights:
    def test_from_iterable_preserves_weights(self) -> None:
        sample = ListSample.from_iterable([1, 2, 3], weights=[0.1, 0.2, 0.3])

        assert sample.weights == [0.1, 0.2, 0.3]

    def test_constructor_rejects_wrong_weight_length(self) -> None:
        with pytest.raises(ValueError, match="Length of weights"):
            ListSample([1, 2, 3], weights=[0.1, 0.2])

    def test_concat_combines_weights(self) -> None:
        left = ListSample([1, 2], weights=[0.1, 0.2])
        right = ListSample([3, 4], weights=[0.3, 0.4])

        result = left.concat(right)

        assert result == [1, 2, 3, 4]
        assert result.weights == [0.1, 0.2, 0.3, 0.4]

    def test_concat_fills_missing_weights_with_ones(self) -> None:
        left = ListSample([1, 2])
        right = ListSample([3, 4], weights=[0.3, 0.4])

        result = left.concat(right)

        assert result == [1, 2, 3, 4]
        assert result.weights == [1.0, 1.0, 0.3, 0.4]
