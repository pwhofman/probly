"""Flax-specific tests for the SWAG method."""

from __future__ import annotations

import pytest

flax = pytest.importorskip("flax")

from flax import nnx  # noqa: E402
import jax  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.method.swag import SWAGPredictor, collect_swag, swag  # noqa: E402
from probly.method.swag.flax import FlaxSWAGPredictor  # noqa: E402
from probly.representer import representer  # noqa: E402
from probly.representer.sampler._common import Sampler  # noqa: E402


@pytest.fixture
def linear_model() -> nnx.Module:
    """Small MLP classifier."""
    rngs = nnx.Rngs(0)
    return nnx.Sequential(
        nnx.Linear(4, 8, rngs=rngs),
        nnx.relu,
        nnx.Linear(8, 3, rngs=rngs),
    )


def make_swag(model: nnx.Module, max_rank: int = 20, scale: float = 0.5) -> FlaxSWAGPredictor:
    """Apply swag and narrow the return type to the flax wrapper."""
    swag_model = swag(model, max_rank=max_rank, scale=scale)
    assert isinstance(swag_model, FlaxSWAGPredictor)
    return swag_model


def weight_vector(model: nnx.Module) -> jax.Array:
    """Flatten the model's parameters into a single vector."""
    weights, _ = ravel_pytree(nnx.state(model, nnx.Param))
    return weights


def random_like(key: jax.Array, reference: jax.Array) -> jax.Array:
    return jax.random.normal(key, reference.shape, reference.dtype)


class TestTransformation:
    """Tests for the swag transformation itself."""

    def test_returns_swag_predictor(self, linear_model: nnx.Module) -> None:
        model = make_swag(linear_model)
        assert isinstance(model, SWAGPredictor)

    def test_original_model_is_not_mutated(self, linear_model: nnx.Module) -> None:
        original = weight_vector(linear_model)
        model = make_swag(linear_model)
        assert model.model is not linear_model

        model.collect(jnp.ones_like(model.mean[...]))
        model.load_mean_parameters()

        assert jnp.array_equal(weight_vector(linear_model), original)
        assert not jnp.array_equal(weight_vector(model.model), original)

    def test_forward_shape(self, linear_model: nnx.Module) -> None:
        model = make_swag(linear_model)
        out = model(jnp.ones((6, 4)))
        assert out.shape == (6, 3)

    def test_statistics_are_not_trainable_params(self, linear_model: nnx.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        params, _ = ravel_pytree(nnx.state(model, nnx.Param))
        assert params.size == model.mean[...].size  # only the wrapped model's weights are nnx.Param


class TestCollect:
    """Tests for the collection of SWAG statistics."""

    def test_moments_match_manual_computation(self, linear_model: nnx.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        snapshots = [random_like(jax.random.key(i), model.mean[...]) for i in range(3)]
        for weights in snapshots:
            model.collect(weights)

        stacked = jnp.stack(snapshots)
        assert int(model.num_collected[...]) == 3
        assert jnp.allclose(model.mean[...], stacked.mean(axis=0), atol=1e-6)
        assert jnp.allclose(model.sq_mean[...], jnp.square(stacked).mean(axis=0), atol=1e-6)

    def test_deviations_keep_last_max_rank_snapshots(self, linear_model: nnx.Module) -> None:
        # The deviation buffer is a ring: snapshot i writes row i % max_rank.
        model = make_swag(linear_model, max_rank=2)
        snapshots = [random_like(jax.random.key(i), model.mean[...]) for i in range(3)]
        for weights in snapshots:
            model.collect(weights)

        stacked = jnp.stack(snapshots)
        assert model.deviations[...].shape[0] == 2
        assert jnp.allclose(model.deviations[...][0], snapshots[2] - stacked.mean(axis=0), atol=1e-6)
        assert jnp.allclose(model.deviations[...][1], snapshots[1] - stacked[:2].mean(axis=0), atol=1e-6)

    def test_collect_swag_dispatch(self, linear_model: nnx.Module) -> None:
        model = make_swag(linear_model)
        collect_swag(model)
        assert int(model.num_collected[...]) == 1
        assert jnp.array_equal(model.mean[...], weight_vector(model.model))


class TestSampling:
    """Tests for sampling weights from the SWAG posterior."""

    def test_sample_before_collect_raises(self, linear_model: nnx.Module) -> None:
        model = make_swag(linear_model)
        with pytest.raises(RuntimeError, match="No weight snapshots have been collected yet"):
            model.sample_parameters()

    def test_sample_with_zero_scale_loads_mean(self, linear_model: nnx.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        for i in range(3):
            model.collect(random_like(jax.random.key(i), model.mean[...]))

        model.sample_parameters(scale=0.0)
        assert jnp.allclose(weight_vector(model.model), model.mean[...], atol=1e-6)

    def test_sample_perturbs_weights(self, linear_model: nnx.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        for i in range(3):
            model.collect(random_like(jax.random.key(i), model.mean[...]))

        model.sample_parameters()
        first = weight_vector(model.model)
        model.sample_parameters()
        second = weight_vector(model.model)
        assert not jnp.array_equal(first, second)

    def test_diagonal_only_sampling(self, linear_model: nnx.Module) -> None:
        model = make_swag(linear_model, max_rank=0)
        for i in range(3):
            model.collect(random_like(jax.random.key(i), model.mean[...]))

        model.sample_parameters()
        assert not jnp.array_equal(weight_vector(model.model), model.mean[...])

    def test_sampling_mode_call_is_stochastic_and_non_mutating(self, linear_model: nnx.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        for i in range(3):
            model.collect(random_like(jax.random.key(i), model.mean[...]))

        before = weight_vector(model.model)
        model.sampling = True
        x = jnp.ones((6, 4))
        first, second = model(x), model(x)
        model.sampling = False
        assert not jnp.array_equal(first, second)
        assert jnp.array_equal(weight_vector(model.model), before)


class TestRepresenter:
    """End-to-end tests through the representer."""

    @pytest.fixture
    def collected_model(self, linear_model: nnx.Module) -> FlaxSWAGPredictor:
        model = make_swag(linear_model, max_rank=5)
        for i in range(4):
            model.collect(random_like(jax.random.key(i), model.mean[...]))
        return model

    def test_representer_dispatches_to_sampler(self, collected_model: FlaxSWAGPredictor) -> None:
        rep = representer(collected_model, num_samples=7)
        assert isinstance(rep, Sampler)

    def test_represent_draws_distinct_samples(self, collected_model: FlaxSWAGPredictor) -> None:
        rep = representer(collected_model, num_samples=3)
        sample = rep.represent(jnp.ones((5, 4)))
        assert sample.array.shape[sample.sample_axis] == 3
        first, second = jnp.moveaxis(sample.array, sample.sample_axis, 0)[:2]
        assert not jnp.array_equal(first, second)

    def test_sampling_restores_weights_and_flag(self, collected_model: FlaxSWAGPredictor) -> None:
        before = weight_vector(collected_model.model)
        rep = representer(collected_model, num_samples=3)
        rep.represent(jnp.ones((5, 4)))
        assert jnp.array_equal(weight_vector(collected_model.model), before)
        assert collected_model.sampling is False
