"""Fixtures for Sample representations."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from probly.representation.sample.jax import JaxSample


@pytest.fixture
def jax_array_sample_2d() -> JaxSample:
    sample_array = jnp.arange(12).reshape((3, 4))
    sample = JaxSample(sample_array, sample_axis=1)

    return sample
