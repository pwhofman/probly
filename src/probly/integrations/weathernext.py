"""Bindings for WeatherNext ensemble forecast models.

WeatherNext models (https://github.com/google-deepmind/weathernext) are JAX functions that map initial
conditions to an xarray forecast and sample one ensemble member per random key. Wrapping such a function in
:class:`WeatherNextPredictor` makes it a stochastic probly predictor: every call draws a fresh member, and
``representer(predictor, num_samples=...)`` collects an ensemble and returns one
:class:`~probly.representation.sample.array.ArraySample` per forecast variable, ready for quantification.
"""

from __future__ import annotations

from typing import Any, override

import numpy as np

from probly.predictor import RandomPredictor
from probly.representation.sample.array import ArraySample
from probly.representer._representer import Representer, representer

DEFAULT_NUM_SAMPLES = 8


class WeatherNextPredictor:
    """Wraps a WeatherNext forecast function as a stochastic probly predictor.

    Args:
        forecast_fn: Callable ``(seed, *args, **kwargs) -> forecast`` running one ensemble member, where
            ``seed`` is an integer to derive the member's random key from and the returned forecast is an
            xarray dataset (or anything with ``data_vars`` and array-valued variables).
        ensemble_fn: Optional callable ``(seeds, *args, **kwargs) -> forecast`` running all members of an
            ensemble at once (e.g. the pmapped WeatherNext rollout), returning a forecast whose variables
            carry a leading sample dimension. When given, the representer prefers it over ``forecast_fn``.
        seed: Base seed; member ``i`` uses ``seed + i``.
    """

    def __init__(self, forecast_fn: Any = None, ensemble_fn: Any = None, seed: int = 0) -> None:  # noqa: ANN401
        """Initialize the predictor from a member and/or bulk forecast function."""
        if forecast_fn is None and ensemble_fn is None:
            msg = "WeatherNextPredictor requires forecast_fn or ensemble_fn."
            raise ValueError(msg)
        self.forecast_fn = forecast_fn
        self.ensemble_fn = ensemble_fn
        self.seed = seed
        self._num_calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Forecast one freshly drawn ensemble member."""
        seed = self.seed + self._num_calls
        self._num_calls += 1
        if self.forecast_fn is not None:
            return self.forecast_fn(seed, *args, **kwargs)
        return self.ensemble_fn([seed], *args, **kwargs)


RandomPredictor.register(WeatherNextPredictor)


@representer.register(WeatherNextPredictor)
class WeatherNextRepresenter(Representer[Any, Any, Any, Any]):
    """Representer collecting a WeatherNext ensemble into per-variable array samples.

    Args:
        predictor: The wrapped WeatherNext predictor.
        num_samples: Number of ensemble members to forecast.
    """

    predictor: WeatherNextPredictor

    def __init__(self, predictor: WeatherNextPredictor, num_samples: int = DEFAULT_NUM_SAMPLES) -> None:
        """Initialize the representer."""
        super().__init__(predictor)
        self.num_samples = num_samples

    @override
    def represent(self, *args: Any, **kwargs: Any) -> dict[str, ArraySample]:
        """Forecast ``num_samples`` members and stack them into one sample per variable."""
        if self.predictor.ensemble_fn is not None:
            seeds = [self.predictor.seed + i for i in range(self.num_samples)]
            forecast = self.predictor.ensemble_fn(seeds, *args, **kwargs)
            return {
                str(name): ArraySample(np.asarray(forecast[name].data), sample_axis=0) for name in forecast.data_vars
            }
        members = [self.predictor(*args, **kwargs) for _ in range(self.num_samples)]
        return {
            str(name): ArraySample(
                np.stack([np.asarray(member[name].data) for member in members], axis=0), sample_axis=0
            )
            for name in members[0].data_vars
        }


__all__ = [
    "WeatherNextPredictor",
    "WeatherNextRepresenter",
]
