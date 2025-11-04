"""Tests for the Ensemble flax.nnx implementation."""

from __future__ import annotations

import random

from flax import nnx
import jax.numpy as jnp
import pytest

from probly.transformation.ensemble import flax as flx


@pytest.fixture
def sample_input_2batches() -> jnp.ndarray:
    """Sample input for 2D models."""
    return jnp.ones((1, 2))


@pytest.fixture
def sample_input_3batches() -> jnp.ndarray:
    """Sample input for linear conv models."""
    return jnp.ones((1, 3, 2))


@pytest.fixture
def sample_input_4batches() -> jnp.ndarray:
    """Sample input for 4D models."""
    return jnp.ones((1, 4))


@pytest.fixture
def sample_input_20batches() -> jnp.ndarray:
    """Large sample for custom tests."""
    return jnp.ones((1, 20))


@pytest.fixture
def sample_input_large_batches() -> jnp.ndarray:
    """Large sample for custom tests."""
    return jnp.ones((1, 10, 20))


def test_generate_flax_ensemble(flax_model_small_2d_2d: nnx.Module) -> None:
    """Test that ensemble generation returns correct length."""
    len_var = int(random.random() * 10)  # noqa: S311 - pseudo random is fine here
    ensemble = flx.generate_flax_ensemble(flax_model_small_2d_2d, n_members=len_var)
    assert len(ensemble) == len_var


def test_generate_flax_ensemble_returns_type(flax_model_small_2d_2d: nnx.Module) -> None:
    """Test that generate_flax_ensemble returns a list."""
    ensemble = flx.generate_flax_ensemble(flax_model_small_2d_2d, n_members=4)
    assert isinstance(ensemble, list)


def test_generate_same_type(flax_model_small_2d_2d: nnx.Module) -> None:
    """Test that all ensemble members have the same type."""
    ensemble = flx.generate_flax_ensemble(flax_model_small_2d_2d, n_members=4)
    for member in ensemble:
        assert isinstance(member, type(flax_model_small_2d_2d))


def test_empty_nmembers(flax_model_small_2d_2d: nnx.Module) -> None:
    """Test that zero members returns empty list."""
    ensemble = flx.generate_flax_ensemble(flax_model_small_2d_2d, n_members=0)
    assert len(ensemble) == 0


def test_is_nnx_module(flax_model_small_2d_2d: nnx.Module) -> None:
    """Ensure all ensemble members are nnx.Module instances."""
    ensemble = flx.generate_flax_ensemble(flax_model_small_2d_2d, n_members=4)
    for member in ensemble:
        assert isinstance(member, nnx.Module)


def test_with_conv_linear_model(flax_conv_linear_model: nnx.Module) -> None:
    """Test that function works with Conv+Linear models."""
    flx.generate_flax_ensemble(flax_conv_linear_model, n_members=2)
    # Known intermittent error for conv linear model


def test_with_regression_model_2d(
    flax_regression_model_2d: nnx.Module,
    sample_input_4batches: jnp.ndarray,
) -> None:
    """Test ensemble with regression model (2D input)."""
    ensemble = flx.generate_flax_ensemble(flax_regression_model_2d, n_members=3)
    assert len(ensemble) == 3
    for member in ensemble:
        output = member(sample_input_4batches)
        assert output.shape == (1, 2)


def test_with_custom_model(flax_custom_model: nnx.Module) -> None:
    """Test for custom model with sample input."""
    x = jnp.ones((1, 10))
    ensemble = flx.generate_flax_ensemble(flax_custom_model, n_members=2)
    assert len(ensemble) == 2
    for member in ensemble:
        output = member(x)
        assert output.shape == (1, 4)


def test_large_ensemble(flax_model_small_2d_2d: nnx.Module) -> None:
    """Test creation of large ensembles."""
    ensemble = flx.generate_flax_ensemble(flax_model_small_2d_2d, n_members=50)
    assert len(ensemble) == 50


def test_different_input_shapes(flax_model_small_2d_2d: nnx.Module) -> None:
    """Test ensembles with varying batch sizes."""
    ensemble = flx.generate_flax_ensemble(flax_model_small_2d_2d, n_members=2)
    for batch_size in [100, 500, 1000, 100_000]:
        input_data = jnp.ones((batch_size, 2))
        for member in ensemble:
            output = member(input_data)
            assert output.shape == (batch_size, 2)
