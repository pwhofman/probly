"""Tests for the WeatherNext bindings."""

from __future__ import annotations

import numpy as np
import pytest

from probly.integrations.weathernext import WeatherNextPredictor, WeatherNextRepresenter
from probly.predictor import RandomPredictor
from probly.quantification import quantify
from probly.representation.sample.numpy import NumpySample
from probly.representer import representer


class FakeVariable:
    def __init__(self, data: np.ndarray) -> None:
        """Store the variable data."""
        self.data = data


class FakeForecast:
    """Duck-typed stand-in for an xarray forecast dataset."""

    def __init__(self, variables: dict[str, np.ndarray]) -> None:
        """Store the variables."""
        self.data_vars = variables

    def __getitem__(self, name: str) -> FakeVariable:
        """Look up a variable by name."""
        return FakeVariable(self.data_vars[name])

    def isel(self, sample: int) -> FakeForecast:
        """Select one member along the leading sample dimension, like ``xarray.Dataset.isel``."""
        return FakeForecast({name: data[sample] for name, data in self.data_vars.items()})


def forecast_fn(seed: int, inputs: np.ndarray) -> FakeForecast:
    rng = np.random.default_rng(seed)
    return FakeForecast(
        {
            "temperature": inputs + rng.normal(0, 1, size=inputs.shape),
            "wind": rng.normal(0, 1, size=inputs.shape),
        }
    )


def ensemble_fn(seeds: list[int], inputs: np.ndarray) -> FakeForecast:
    members = [forecast_fn(seed, inputs) for seed in seeds]
    return FakeForecast(
        {name: np.stack([member[name].data for member in members], axis=0) for name in members[0].data_vars}
    )


@pytest.fixture
def predictor() -> WeatherNextPredictor:
    return WeatherNextPredictor(forecast_fn, seed=100)


def test_is_random_predictor(predictor: WeatherNextPredictor) -> None:
    assert isinstance(predictor, RandomPredictor)


def test_calls_draw_fresh_members(predictor: WeatherNextPredictor) -> None:
    inputs = np.zeros((2, 3))
    first = predictor(inputs)
    second = predictor(inputs)
    assert not np.array_equal(first["temperature"].data, second["temperature"].data)


def test_representer_returns_per_variable_samples(predictor: WeatherNextPredictor) -> None:
    rep = representer(predictor, num_samples=5)
    assert isinstance(rep, WeatherNextRepresenter)

    samples = rep.represent(np.zeros((2, 3)))
    assert set(samples) == {"temperature", "wind"}
    for sample in samples.values():
        assert isinstance(sample, NumpySample)
        assert sample.array.shape == (5, 2, 3)
        assert sample.sample_axis == 0


def test_quantify_on_represented_samples(predictor: WeatherNextPredictor) -> None:
    samples = representer(predictor, num_samples=6).represent(np.zeros((2, 3)))
    total = quantify(samples["temperature"])["total"]
    assert np.asarray(total).shape == (2, 3)
    assert float(np.asarray(total).mean()) > 0.0


def test_members_are_reproducible_per_seed() -> None:
    inputs = np.zeros((2, 2))
    first = WeatherNextPredictor(forecast_fn, seed=7)(inputs)
    second = WeatherNextPredictor(forecast_fn, seed=7)(inputs)
    np.testing.assert_array_equal(first["temperature"].data, second["temperature"].data)


def test_requires_a_forecast_function() -> None:
    with pytest.raises(ValueError, match="requires forecast_fn or ensemble_fn"):
        WeatherNextPredictor()


def test_ensemble_fn_is_preferred_for_bulk_representation() -> None:
    predictor = WeatherNextPredictor(ensemble_fn=ensemble_fn, seed=3)
    samples = representer(predictor, num_samples=4).represent(np.zeros((2, 3)))
    assert samples["temperature"].array.shape == (4, 2, 3)
    assert samples["temperature"].sample_axis == 0
    # Matches the sequential path member-for-member thanks to the shared seed convention.
    sequential = WeatherNextPredictor(forecast_fn, seed=3)(np.zeros((2, 3)))
    np.testing.assert_array_equal(samples["temperature"].array[0], sequential["temperature"].data)


def test_single_call_with_ensemble_fn_returns_one_member() -> None:
    inputs = np.zeros((2, 3))
    member = WeatherNextPredictor(ensemble_fn=ensemble_fn, seed=3)(inputs)
    # Same shape and values as the member-wise path: no leading sample axis of length one.
    assert member["temperature"].data.shape == (2, 3)
    np.testing.assert_array_equal(member["temperature"].data, forecast_fn(3, inputs)["temperature"].data)
