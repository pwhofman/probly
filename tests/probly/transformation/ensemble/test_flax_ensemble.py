"""Test ensemble flax.py."""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("flax")
from flax import linen as nn
import jax
import jax.numpy as jnp

from probly.transformation.ensemble.flax import FlaxEnsemble, generate_flax_ensemble


class SimpleFlaxModel(nn.Module):
    features: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, _train: bool = False) -> jnp.ndarray:  # _train kept for **call_kwargs test
        return nn.Dense(self.features)(x)


def test_ensemble_init() -> None:
    n_members = 3
    base_model = SimpleFlaxModel()
    ensemble_model = generate_flax_ensemble(base_model, n_members)
    assert isinstance(ensemble_model, FlaxEnsemble)
    assert ensemble_model.n_members == n_members


def test_ensemble_attributes() -> None:
    # tests structural attributes e.g. n_members, base_module
    base_model = SimpleFlaxModel()
    model = generate_flax_ensemble(base_model, n_members=3)

    assert isinstance(model, FlaxEnsemble)
    assert model.n_members == 3
    assert model.base_module is SimpleFlaxModel


def test_generate_distinct_ensembles() -> None:
    # makes sure generator creates distinct ensemble instances and base_module is saved properly
    base_model = SimpleFlaxModel()
    e1 = generate_flax_ensemble(base_model, n_members=2)
    e2 = generate_flax_ensemble(base_model, n_members=2)

    assert e1 is not e2
    assert e1.base_module is e2.base_module is SimpleFlaxModel


@pytest.mark.parametrize("n_members", [1, 3, 5])
def test_ensemble_different_sizes(n_members: int) -> None:
    base_model = SimpleFlaxModel()
    model = generate_flax_ensemble(base_model, n_members)
    assert model.n_members == n_members


def test_call_signature_accepts_return_all_and_kwargs() -> None:
    sig = inspect.signature(FlaxEnsemble.__call__)
    params = sig.parameters
    assert "return_all" in params
    # make sure it accepts **kwargs (VAR_KEYWORD) for call kwargs
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_return_all_and_average_shape_and_values() -> None:
    # makesure generator stores n_members and base module for later runtime (sprint2 changes)
    base_model = SimpleFlaxModel()
    model = generate_flax_ensemble(base_model, n_members=4)
    assert isinstance(model, FlaxEnsemble)
    assert model.n_members == 4
    assert model.base_module is SimpleFlaxModel


class TupleOutModel(nn.Module):
    features: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        a = nn.Dense(self.features)(x)
        b = nn.Dense(self.features)(x) + 1.0
        return {"a": a, "b": b}


def test_pytree_output_stacking() -> None:
    # gener. accepts a model that makes PyTree output
    base_model = TupleOutModel()
    model = generate_flax_ensemble(base_model, n_members=3)
    assert isinstance(model, FlaxEnsemble)
    assert model.base_module is TupleOutModel


def test_return_all_and_average_shape_and_values_runtime() -> None:
    # makes sure ensemble stacks per member output properl. and in proper form and averages correctly at runtime
    rng = jax.random.PRNGKey(1)
    x = jnp.ones((3, 4))
    base_model = SimpleFlaxModel()
    model = generate_flax_ensemble(base_model, n_members=4)
    variables = model.init(rng, x)

    all_out = model.apply(variables, x, return_all=True)
    assert all_out.shape == (x.shape[0], 4, base_model.features)

    avg_out = model.apply(variables, x, return_all=False)
    computed_mean = jnp.mean(all_out, axis=1)
    assert jnp.allclose(avg_out, computed_mean)


def test_pytree_output_stacking_runtime() -> None:
    # explicit test for PyTree output stacking at runtime (transformation notebook)
    class DictOutModel(nn.Module):
        features: int = 2

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
            return {"a": nn.Dense(self.features)(x), "b": nn.Dense(self.features)(x) + 1.0}

    rng = jax.random.PRNGKey(2)
    x = jnp.ones((2, 2))
    model = generate_flax_ensemble(DictOutModel(), n_members=3)
    variables = model.init(rng, x)

    stacked = model.apply(variables, x, return_all=True)
    assert set(stacked.keys()) == {"a", "b"}
    assert stacked["a"].shape == (2, 3, 2)
    assert stacked["b"].shape == (2, 3, 2)


def test_generate_from_class() -> None:
    # make sure gen from class works
    model = generate_flax_ensemble(SimpleFlaxModel, n_members=2)
    assert isinstance(model, FlaxEnsemble)
    assert model.n_members == 2


def test_generate_with_non_module_input_no_validation() -> None:
    model = generate_flax_ensemble("not a module", n_members=2)
    assert isinstance(model, FlaxEnsemble)
    assert model.n_members == 2
    # base_module becomes str since input was string instance
    assert model.base_module is str


def test_dataclass_kwargs_doesnt_crash() -> None:
    base_model = SimpleFlaxModel(features=7)
    model = generate_flax_ensemble(base_model, n_members=2)
    # gener. should get constructor kwargs from dataclass-like modules(if possib.)
    assert model.base_module is SimpleFlaxModel
    if model.base_kwargs is not None:
        assert model.base_kwargs.get("features") == 7


@pytest.mark.parametrize(
    "fixture_name",
    [
        "flax_model_small_2d_2d",
        "flax_conv_linear_model",
        "flax_regression_model_1d",
        "flax_regression_model_2d",
        "flax_custom_model",
    ],
)
def test_generate_with_fixture_models(request: pytest.FixtureRequest, fixture_name: str) -> None:
    # Ensure ensemble generation works with models provided by tests/probly/fixtures/flax_models.py
    model_instance = request.getfixturevalue(fixture_name)
    ens = generate_flax_ensemble(model_instance, n_members=3)
    assert isinstance(ens, FlaxEnsemble)
    assert ens.n_members == 3
    assert ens.base_module is model_instance.__class__
