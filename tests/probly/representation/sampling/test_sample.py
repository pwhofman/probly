"""Test sample dispatching logic."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("jax")
from jax import numpy as jnp
import numpy as np
import torch

from probly.representation.sample import ArraySample, create_sample
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
