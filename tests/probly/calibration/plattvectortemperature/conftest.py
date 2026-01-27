"""Conftest for isotonic regression tests containing necessary fixtures."""

from __future__ import annotations

from collections.abc import Iterable

from flax import nnx
import jax
import pytest
import torch
from torch import nn

JaxDataLoader = Iterable[tuple[jax.Array, jax.Array]]
SetupReturnType = tuple[nnx.Module, nnx.Module, JaxDataLoader, Iterable[tuple[jax.Array, jax.Array]]]

@pytest.fixture
def flax_binary_model():
    return nnx.Sequential(
        nnx.Linear(2, 1, rngs=nnx.Rngs(0)),
    )

@pytest.fixture
def flax_multiclass_model():
    return nnx.Sequential(
        nnx.Linear(2, 3, rngs=nnx.Rngs(0)),
    )

@pytest.fixture
def torch_binary_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(10, 1),
    )

@pytest.fixture
def torch_multiclass_model():
    return nn.Sequential(
        nn.Linear(2, 3),
    )


@pytest.fixture
def flax_setup_binary(flax_binary_model: nnx.Module):
    base = flax_binary_model

    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (20, 2))

    key, subkey = jax.random.split(key)
    labels = jax.random.randint(subkey, (20,), minval=0, maxval=1)

    return base, inputs, labels

@pytest.fixture
def flax_setup_multiclass(flax_multiclass_model: nnx.Module):
    base = flax_multiclass_model

    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (20, 2))

    key, subkey = jax.random.split(key)
    labels = jax.random.randint(subkey, (20,), minval=0, maxval=3)

    return base, inputs, labels

@pytest.fixture
def torch_setup_multiclass(torch_multiclass_model: nn.Module):
    base = torch_multiclass_model

    inputs = torch.randn(20, 2)
    labels = torch.randint(0, 3, (20,))

    return base, inputs, labels
