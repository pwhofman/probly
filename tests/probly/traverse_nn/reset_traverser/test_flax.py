from __future__ import annotations

from collections.abc import Callable

import pytest

from probly.traverse_nn import nn_compose, reset_traverser
from pytraverse import CLONE, traverse

flax = pytest.importorskip("flax")

from flax import nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.layers.flax import DropConnectLinear  # noqa: E402
from probly.traverse_nn.reset_traverser.flax import _RESET_SEED, RNGS  # noqa: E402


def reset[T: nnx.Module](model: T, init: dict | None = None) -> T:
    """Clone ``model`` and reset the parameters of the clone."""
    return traverse(model, nn_compose(reset_traverser), init={CLONE: True, **(init or {})})


def param_shapes(model: nnx.Module) -> list[tuple[int, ...]]:
    """Return the shapes of every parameter in ``model``, in a stable order."""
    return [jnp.shape(v) for v in jax.tree.leaves(nnx.state(model, nnx.Param))]


def kernels(model: nnx.Module) -> list[jnp.ndarray]:
    """Return the kernels of every linear layer in ``model``, in traversal order."""
    return [jnp.asarray(m.kernel[...]) for _, m in nnx.iter_modules(model) if isinstance(m, nnx.Linear)]


class ResettableLinear(nnx.Linear):
    """A linear layer that resets itself in place, and tracks how often it was asked to."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs) -> None:
        """Initialize the layer and give it its own rng stream."""
        super().__init__(in_features, out_features, rngs=rngs)
        self.rng_collection = "resettable"
        self.rngs = nnx.Rngs(0)["resettable"].fork()
        self.reset_count = 0

    def reset_parameters(self) -> None:
        """Zero the kernel so the reset is unmistakable, and count the call."""
        self.kernel = nnx.Param(jnp.zeros_like(self.kernel[...]))
        self.reset_count += 1


class TestResetParameters:
    def test_parameters_change(self, flax_model_small_2d_2d: nnx.Module) -> None:
        before = kernels(flax_model_small_2d_2d)

        after = kernels(reset(flax_model_small_2d_2d))

        assert all(not bool(jnp.array_equal(b, a)) for b, a in zip(before, after, strict=True))

    def test_original_model_is_left_untouched(self, flax_model_small_2d_2d: nnx.Module) -> None:
        before = kernels(flax_model_small_2d_2d)

        reset(flax_model_small_2d_2d)

        assert all(bool(jnp.array_equal(b, a)) for b, a in zip(before, kernels(flax_model_small_2d_2d), strict=True))

    def test_shapes_are_preserved(self, flax_model_small_2d_2d: nnx.Module) -> None:
        before = kernels(flax_model_small_2d_2d)

        after = kernels(reset(flax_model_small_2d_2d))

        assert [k.shape for k in after] == [k.shape for k in before]

    def test_model_stays_callable(self, flax_regression_model_2d: nnx.Module) -> None:
        model = reset(flax_regression_model_2d)

        assert model(jnp.ones((3, 4))).shape == (3, 2)

    def test_custom_module_type_is_preserved(self, flax_custom_model: nnx.Module) -> None:
        model = reset(flax_custom_model)

        assert type(model) is type(flax_custom_model)
        assert model(jnp.ones((1, 10))).shape == (1, 4)

    @pytest.mark.parametrize(
        "layer_factory",
        [
            lambda rngs: nnx.Linear(3, 4, rngs=rngs),
            lambda rngs: nnx.Conv(3, 4, (3, 3), rngs=rngs),
            lambda rngs: nnx.BatchNorm(4, rngs=rngs),
            lambda rngs: nnx.LayerNorm(4, rngs=rngs),
        ],
        ids=["linear", "conv", "batchnorm", "layernorm"],
    )
    def test_stock_flax_layers_are_reconstructed(
        self,
        layer_factory: Callable[[nnx.Rngs], nnx.Module],
        flax_rngs: nnx.Rngs,
    ) -> None:
        layer = layer_factory(flax_rngs)

        new_layer = reset(layer)

        assert type(new_layer) is type(layer)
        assert param_shapes(new_layer) == param_shapes(layer)

    def test_parameterless_layers_pass_through(self, flax_rngs: nnx.Rngs) -> None:
        model = nnx.Sequential(nnx.Dropout(rate=0.5, deterministic=True, rngs=flax_rngs), nnx.relu)

        assert reset(model)(jnp.ones((1, 2))).shape == (1, 2)

    def test_activation_functions_survive_the_traversal(self, flax_rngs: nnx.Rngs) -> None:
        """``nnx.Sequential`` holds bare callables next to its layers; they must pass through."""
        model = nnx.Sequential(nnx.Linear(4, 4, rngs=flax_rngs), nnx.relu, nnx.Linear(4, 2, rngs=flax_rngs))

        new_model = reset(model)

        assert any(not isinstance(layer, nnx.Module) for layer in new_model.layers)
        assert new_model(jnp.ones((1, 4))).shape == (1, 2)


class TestResetParametersHook:
    def test_reset_parameters_is_preferred_over_reconstruction(self, flax_rngs: nnx.Rngs) -> None:
        layer = ResettableLinear(2, 2, rngs=flax_rngs)

        new_layer = reset(layer)

        assert new_layer.reset_count == 1
        assert bool(jnp.array_equal(new_layer.kernel[...], jnp.zeros((2, 2))))

    def test_rng_stream_is_reforked(self, flax_rngs: nnx.Rngs) -> None:
        layer = ResettableLinear(2, 2, rngs=flax_rngs)
        before = jax.random.key_data(layer.rngs.key[...])

        new_layer = reset(layer)

        after = jax.random.key_data(new_layer.rngs.key[...])
        assert not bool(jnp.array_equal(before, after))


class TestRngs:
    def test_consecutive_resets_draw_different_parameters(self, flax_model_small_2d_2d: nnx.Module) -> None:
        """The property ensembles rely on: cloning-and-resetting N times gives N distinct models."""
        first = kernels(reset(flax_model_small_2d_2d))
        second = kernels(reset(flax_model_small_2d_2d))

        assert all(not bool(jnp.array_equal(a, b)) for a, b in zip(first, second, strict=True))

    def test_explicit_rngs_makes_resets_reproducible(self, flax_model_small_2d_2d: nnx.Module) -> None:
        first = kernels(reset(flax_model_small_2d_2d, {RNGS: nnx.Rngs(7)}))
        second = kernels(reset(flax_model_small_2d_2d, {RNGS: nnx.Rngs(7)}))

        assert all(bool(jnp.array_equal(a, b)) for a, b in zip(first, second, strict=True))

    def test_integer_seeds_are_accepted(self, flax_model_small_2d_2d: nnx.Module) -> None:
        first = kernels(reset(flax_model_small_2d_2d, {RNGS: 7}))
        second = kernels(reset(flax_model_small_2d_2d, {RNGS: nnx.Rngs(7)}))

        assert all(bool(jnp.array_equal(a, b)) for a, b in zip(first, second, strict=True))

    def test_default_stream_does_not_reproduce_a_conventionally_seeded_model(self) -> None:
        """A default stream starting at seed 0 would make the first reset of the process a no-op.

        Checked against a fresh copy of the default stream rather than the shared one, so the
        assertion does not depend on how many resets ran before it.
        """
        layer = nnx.Linear(2, 2, rngs=nnx.Rngs(0))

        new_layer = reset(layer, {RNGS: nnx.Rngs(_RESET_SEED)})

        assert not bool(jnp.array_equal(new_layer.kernel[...], layer.kernel[...]))

    def test_layers_of_one_model_draw_from_a_shared_stream(self, flax_model_small_2d_2d: nnx.Module) -> None:
        """All three layers have the same shape; a per-layer stream would give them equal kernels."""
        first, second, third = kernels(reset(flax_model_small_2d_2d, {RNGS: 7}))

        assert not bool(jnp.array_equal(first, second))
        assert not bool(jnp.array_equal(second, third))


class TestUnsupportedLayers:
    def test_layer_that_cannot_be_reconstructed_raises(self, flax_rngs: nnx.Rngs) -> None:
        layer = DropConnectLinear(nnx.Linear(2, 2, rngs=flax_rngs), rngs=flax_rngs)

        with pytest.raises(NotImplementedError, match="base_layer"):
            reset(layer)


class TestEnsembleIntegration:
    def test_ensemble_members_differ_when_parameters_are_reset(self, flax_model_small_2d_2d: nnx.Module) -> None:
        from probly.transformation.ensemble import ensemble  # noqa: PLC0415

        members = [kernels(m)[0] for m in ensemble(flax_model_small_2d_2d, num_members=3, reset_params=True)]

        assert all(
            not bool(jnp.array_equal(members[i], members[j]))
            for i in range(len(members))
            for j in range(i + 1, len(members))
        )

    def test_ensemble_members_are_identical_without_reset(self, flax_model_small_2d_2d: nnx.Module) -> None:
        from probly.transformation.ensemble import ensemble  # noqa: PLC0415

        members = [kernels(m)[0] for m in ensemble(flax_model_small_2d_2d, num_members=3, reset_params=False)]

        assert all(bool(jnp.array_equal(members[0], m)) for m in members)
