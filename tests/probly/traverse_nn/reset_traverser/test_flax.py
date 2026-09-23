from __future__ import annotations

from collections.abc import Callable
import subprocess
import sys
import textwrap

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


class RandomResettableLinear(ResettableLinear):
    """A custom reset hook that draws weights from its stored RNG stream."""

    def reset_parameters(self) -> None:
        self.kernel[...] = jax.random.normal(self.rngs(), self.kernel.shape)
        self.reset_count += 1


class ContainerResettableLinear(nnx.Linear):
    """A custom layer using named streams from an Rngs container."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs) -> None:
        """Initialize the layer and retain its named RNG container."""
        super().__init__(in_features, out_features, rngs=rngs)
        self.rngs = rngs

    def reset_parameters(self) -> None:
        self.kernel[...] = jax.random.normal(self.rngs.params(), self.kernel.shape)
        assert self.bias is not None
        self.bias[...] = jax.random.normal(self.rngs.noise(), self.bias.shape)


class Tied(nnx.Module):
    """A model that applies one linear layer under two attribute names, as weight-tied models do."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        """Create the input layer and the tied layer."""
        self.inp = nnx.Linear(4, 4, rngs=rngs)
        self.a = nnx.Linear(4, 4, rngs=rngs)
        self.b = self.a

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.b(nnx.relu(self.a(nnx.relu(self.inp(x)))))


class NoisyHead(nnx.Module):
    """A module that owns no variables itself, but keeps an RNG container to draw noise from."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        """Create the linear layer and the noise stream."""
        self.linear = nnx.Linear(2, 2, rngs=rngs)
        self.rngs = nnx.Rngs(noise=5)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x) + jax.random.normal(self.rngs.noise(), x.shape)


def dropout_model() -> nnx.Sequential:
    """Return a linear layer followed by a dropout layer with a stream of its own."""
    return nnx.Sequential(nnx.Linear(8, 64, rngs=nnx.Rngs(0)), nnx.Dropout(0.5, rngs=nnx.Rngs(dropout=0)))


def layer_of[T](model: nnx.Sequential, index: int, cls: type[T]) -> T:
    """Return the layer at ``index`` of a sequential model, checking its type."""
    layer = model.layers[index]
    assert isinstance(layer, cls)
    return layer


def dropout_stream(model: nnx.Sequential) -> nnx.RngStream:
    """Return the stream of the dropout layer of a :func:`dropout_model`."""
    stream = layer_of(model, 1, nnx.Dropout).rngs
    assert isinstance(stream, nnx.RngStream)
    return stream


def stream_key(stream: nnx.RngStream) -> jax.Array:
    """Return the raw key data of an RNG stream."""
    return jax.random.key_data(stream.key[...])


def dropped(model: nnx.Module, x: jax.Array) -> jax.Array:
    """Return where the model output was zeroed by dropout."""
    return model(x) == 0


@pytest.mark.parametrize("first_layer", ["activation", "dropout"])
@pytest.mark.parametrize("entrypoint", ["ensemble", "traverse"])
def test_reset_lazy_loading_before_first_child(first_layer: str, entrypoint: str) -> None:
    script = textwrap.dedent(
        """
        import sys
        from flax import nnx
        import jax.numpy as jnp
        from probly.transformation.ensemble import ensemble
        from probly.traverse_nn import nn_compose, reset_traverser
        from pytraverse import CLONE, traverse

        first_layer, entrypoint = sys.argv[1:]
        rngs = nnx.Rngs(0)
        first = nnx.relu if first_layer == "activation" else nnx.Dropout(
            rate=0.5, deterministic=True, rngs=rngs
        )
        model = nnx.Sequential(first, nnx.Linear(2, 2, rngs=rngs))
        before = model.layers[1].kernel[...]
        assert "probly.traverse_nn.reset_traverser.flax" not in sys.modules
        assert traverse(nnx.relu, reset_traverser) is nnx.relu
        assert "probly.traverse_nn.reset_traverser.flax" not in sys.modules

        if entrypoint == "ensemble":
            members = list(ensemble(model, num_members=2, reset_params=True))
        else:
            # Importing traversal alone must not load reset support.
            from probly.traverse_nn.flax import flax_traverser
            assert "probly.traverse_nn.reset_traverser.flax" not in sys.modules
            members = [
                traverse(model, nn_compose(reset_traverser, nn_traverser=flax_traverser), init={CLONE: True})
                for _ in range(2)
            ]

        assert "probly.traverse_nn.reset_traverser.flax" in sys.modules
        assert all(member(jnp.ones((1, 2))).shape == (1, 2) for member in members)
        assert not jnp.array_equal(members[0].layers[1].kernel[...], members[1].layers[1].kernel[...])
        assert jnp.array_equal(model.layers[1].kernel[...], before)
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed script and parametrized test cases
        [sys.executable, "-c", script, first_layer, entrypoint],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
    @pytest.mark.parametrize("named_streams", [False, True])
    def test_rng_container_interface_and_seeding_are_preserved(self, named_streams: bool) -> None:
        rngs = nnx.Rngs(params=0, noise=1) if named_streams else nnx.Rngs(0)
        layer = ContainerResettableLinear(2, 2, rngs=rngs)
        original_kernel = layer.kernel[...]
        original_counts = {name: stream.count[...] for name, stream in rngs.items()}
        first = reset(layer, {RNGS: 7})
        repeated = reset(layer, {RNGS: 7})
        other = reset(layer, {RNGS: 99})

        assert isinstance(first.rngs, nnx.Rngs)
        assert set(first.rngs) == set(rngs)
        assert first.bias is not None
        assert repeated.bias is not None
        assert other.bias is not None
        assert jnp.array_equal(first.kernel[...], repeated.kernel[...])
        assert jnp.array_equal(first.bias[...], repeated.bias[...])
        assert not jnp.array_equal(first.kernel[...], other.kernel[...])
        assert not jnp.array_equal(first.bias[...], other.bias[...])
        first.reset_parameters()  # Named streams remain usable after the reset.
        assert jnp.array_equal(layer.kernel[...], original_kernel)
        for name, stream in rngs.items():
            assert jnp.array_equal(stream.count[...], original_counts[name])

    def test_single_rng_stream_retains_its_collection(self) -> None:
        layer = RandomResettableLinear(2, 2, rngs=nnx.Rngs(0))
        layer.rngs = nnx.Rngs(params=0).params.fork()
        del layer.rng_collection
        result = reset(layer, {RNGS: nnx.Rngs(params=7)})
        assert isinstance(result.rngs, nnx.RngStream)
        assert result.rngs.tag == "params"
        assert not jnp.array_equal(result.kernel[...], layer.kernel[...])

    def test_random_hook_uses_explicit_reset_seed(self) -> None:
        layer = RandomResettableLinear(2, 2, rngs=nnx.Rngs(0))
        first = reset(layer, {RNGS: 7})
        repeated = reset(layer, {RNGS: nnx.Rngs(7)})
        other = reset(layer, {RNGS: 99})

        assert jnp.array_equal(first.kernel[...], repeated.kernel[...])
        assert not jnp.array_equal(first.kernel[...], other.kernel[...])
        expected_stream = nnx.Rngs(7)["resettable"].fork()
        expected_kernel = jax.random.normal(expected_stream(), (2, 2))
        assert jnp.array_equal(first.kernel[...], expected_kernel)
        # The stored stream must retain the draw consumed by the hook.
        assert jnp.array_equal(first.rngs(), expected_stream())
        assert first.reset_count == 1

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


class TestSharedModules:
    """A module referenced under several names is reset once, and stays one module."""

    def test_tied_layer_stays_tied(self, flax_rngs: nnx.Rngs) -> None:
        model = Tied(flax_rngs)

        new_model = reset(model)

        assert new_model.a is new_model.b
        assert not bool(jnp.array_equal(new_model.a.kernel[...], model.a.kernel[...]))
        # nnx stores a shared module once, so the reset model has as many parameters as the original.
        assert param_shapes(new_model) == param_shapes(model)

    def test_layer_reused_in_a_sequential_stays_shared(self, flax_rngs: nnx.Rngs) -> None:
        shared = nnx.Linear(4, 4, rngs=flax_rngs)
        model = nnx.Sequential(nnx.Linear(4, 4, rngs=flax_rngs), nnx.relu, shared, nnx.relu, shared)

        new_model = reset(model)

        first, reused = layer_of(new_model, 0, nnx.Linear), layer_of(new_model, 2, nnx.Linear)
        assert layer_of(new_model, 4, nnx.Linear) is reused
        assert not bool(jnp.array_equal(reused.kernel[...], shared.kernel[...]))
        x = jnp.ones((1, 4))
        assert bool(jnp.array_equal(new_model(x), reused(nnx.relu(reused(nnx.relu(first(x)))))))

    def test_shared_layer_is_reset_once(self) -> None:
        """The number of draws does not depend on how many attributes refer to a layer."""
        layer = ResettableLinear(2, 2, rngs=nnx.Rngs(0))

        new_model = reset(nnx.Sequential(layer, layer))

        new_layer = layer_of(new_model, 0, ResettableLinear)
        assert layer_of(new_model, 1, ResettableLinear) is new_layer
        assert new_layer.reset_count == 1

    def test_distinct_layers_with_equal_weights_stay_distinct(self) -> None:
        model = nnx.Sequential(nnx.Linear(4, 4, rngs=nnx.Rngs(0)), nnx.Linear(4, 4, rngs=nnx.Rngs(0)))

        new_model = reset(model)

        first, second = layer_of(new_model, 0, nnx.Linear), layer_of(new_model, 1, nnx.Linear)
        assert first is not second
        assert not bool(jnp.array_equal(first.kernel[...], second.kernel[...]))

    def test_consecutive_resets_of_a_tied_model_are_independent(self, flax_rngs: nnx.Rngs) -> None:
        model = Tied(flax_rngs)

        first = reset(model)
        second = reset(model)

        assert first.a is first.b
        assert second.a is second.b
        assert first.a is not second.a
        assert not bool(jnp.array_equal(first.a.kernel[...], second.a.kernel[...]))


class TestRngStreams:
    """Modules without variables of their own, such as ``nnx.Dropout``, may still keep an RNG stream."""

    def test_dropout_stream_is_reforked(self) -> None:
        model = dropout_model()
        stream = dropout_stream(model)
        key, count = stream_key(stream), stream.count[...]

        new_stream = dropout_stream(reset(model))

        assert not bool(jnp.array_equal(stream_key(new_stream), key))
        # The stream of the original model is left untouched.
        assert bool(jnp.array_equal(stream_key(stream), key))
        assert bool(jnp.array_equal(stream.count[...], count))

    def test_reset_models_draw_different_dropout_masks(self) -> None:
        model = dropout_model()
        x = jnp.ones((1, 8))

        masks = [dropped(model, x)] + [dropped(reset(model), x) for _ in range(3)]

        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                assert not bool(jnp.array_equal(masks[i], masks[j])), (i, j)

    def test_explicit_rngs_make_dropout_streams_reproducible(self) -> None:
        model = dropout_model()

        first = stream_key(dropout_stream(reset(model, {RNGS: 7})))
        repeated = stream_key(dropout_stream(reset(model, {RNGS: nnx.Rngs(7)})))
        other = stream_key(dropout_stream(reset(model, {RNGS: 8})))

        assert bool(jnp.array_equal(first, repeated))
        assert not bool(jnp.array_equal(first, other))

    def test_reforked_dropout_stream_keeps_its_tag(self) -> None:
        """``nnx.reseed`` finds streams by tag, so a re-forked stream must still be a ``dropout`` stream."""
        model = dropout_model()
        x = jnp.ones((1, 8))
        first, second = reset(model), reset(model)

        assert dropout_stream(first).tag == "dropout"
        assert not bool(jnp.array_equal(dropped(first, x), dropped(second, x)))
        nnx.reseed(first, dropout=3)
        nnx.reseed(second, dropout=3)
        assert bool(jnp.array_equal(dropped(first, x), dropped(second, x)))

    def test_dropout_without_a_stream_is_left_alone(self) -> None:
        model = nnx.Sequential(nnx.Linear(8, 64, rngs=nnx.Rngs(0)), nnx.Dropout(0.5))

        dropout = layer_of(reset(model), 1, nnx.Dropout)

        assert dropout.rngs is None
        out = dropout(jnp.ones((1, 64)), rngs=nnx.Rngs(dropout=0))
        assert set(jnp.unique(out).tolist()) <= {0.0, 2.0}

    def test_rng_container_of_a_module_without_variables_is_reforked(self) -> None:
        model = NoisyHead(nnx.Rngs(0))
        key = stream_key(model.rngs.noise)

        new_model = reset(model)

        assert isinstance(new_model.rngs, nnx.Rngs)
        assert set(new_model.rngs) == {"noise"}
        assert not bool(jnp.array_equal(stream_key(new_model.rngs.noise), key))
        assert bool(jnp.array_equal(stream_key(model.rngs.noise), key))


class TestUnsupportedLayers:
    def test_layer_that_cannot_be_reconstructed_raises(self, flax_rngs: nnx.Rngs) -> None:
        layer = DropConnectLinear(nnx.Linear(2, 2, rngs=flax_rngs), rngs=flax_rngs)

        with pytest.raises(NotImplementedError, match="base_layer"):
            reset(layer)


class TestEnsembleIntegration:
    def test_rng_containers_produce_distinct_members(self) -> None:
        from probly.transformation.ensemble import ensemble  # noqa: PLC0415

        layer = ContainerResettableLinear(2, 2, rngs=nnx.Rngs(params=0, noise=1))
        members = list(ensemble(layer, num_members=3, reset_params=True))
        assert all(isinstance(member.rngs, nnx.Rngs) for member in members)
        assert all(
            not jnp.array_equal(members[i].kernel[...], members[j].kernel[...])
            for i in range(len(members))
            for j in range(i + 1, len(members))
        )

    def test_random_reset_hooks_produce_distinct_members(self) -> None:
        from probly.transformation.ensemble import ensemble  # noqa: PLC0415

        layer = RandomResettableLinear(2, 2, rngs=nnx.Rngs(0))
        original_kernel = layer.kernel[...]
        original_key = jax.random.key_data(layer.rngs.key[...])
        original_count = layer.rngs.count[...]
        members = list(ensemble(layer, num_members=3, reset_params=True))

        assert all(member.reset_count == 1 for member in members)
        assert all(
            not jnp.array_equal(members[i].kernel[...], members[j].kernel[...])
            for i in range(len(members))
            for j in range(i + 1, len(members))
        )
        assert jnp.array_equal(layer.kernel[...], original_kernel)
        assert jnp.array_equal(jax.random.key_data(layer.rngs.key[...]), original_key)
        assert jnp.array_equal(layer.rngs.count[...], original_count)
        assert layer.reset_count == 0

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
