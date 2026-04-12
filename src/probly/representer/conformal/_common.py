"""This module contains common code for conformal representers."""

from __future__ import annotations

from typing import Any, override

import numpy as np

from lazy_dispatch.singledispatch import lazydispatch
from probly.conformal.utils import is_conformal_calibrated
from probly.predictor._common import (
    ConformalClassificationPredictor,
    ConformalPredictor,
    ConformalQuantileRegressionPredictor,
    ConformalRegressionPredictor,
    predict_raw,
)
from probly.representation.conformal_set._common import (
    ConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from probly.representation.sample import create_sample
from probly.representation.sample._common import Sample
from probly.representation.sample.array import ArraySample
from probly.representer import representer
from probly.representer._representer import Representer


class ConformalRepresenter[**In, Out: ConformalSet](Representer[Any, In, Out, ConformalSet]):
    def __init__(self, predictor: ConformalPredictor[In, Out]) -> None:
        super().__init__(predictor)


@representer.register(ConformalClassificationPredictor)
class ConformalClassificationRepresenter[**In, Out: OneHotConformalSet](ConformalRepresenter[In, Out]):
    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> OneHotConformalSet:
        """Predict for a conformal classification predictor."""
        if not is_conformal_calibrated(self.predictor):
            msg = "The model must first be calibrated before it can be represented."
            raise ValueError(msg)
        predictions = predict_raw(self.predictor, *args, **kwargs)
        non_conformity_score = self.predictor.non_conformity_score.compute(predictions)
        set_prediction = non_conformity_score <= self.predictor.conformal_quantile
        return create_onehot_conformal_set(set_prediction)


@lazydispatch
def flatten_sample[T](sample: Sample[T]) -> T:
    msg = f"Flattening not implemented for type {type(sample)}."
    raise NotImplementedError(msg)


@flatten_sample.register(ArraySample)
def flatten_sample_numpy(sample: ArraySample) -> np.ndarray:
    raw_array = sample.array
    if raw_array.ndim == 1:
        return raw_array
    if raw_array.ndim == 2 and (raw_array.shape[1] == 1 or raw_array.shape[0] == 1):
        return raw_array.flatten()
    return sample.sample_mean()


@representer.register(ConformalRegressionPredictor)
class ConformalRegressionRepresenter[**In, Out: ConformalSet](ConformalRepresenter[In, Out]):
    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> ConformalSet:
        """Predict for a conformal regression predictor."""
        if not is_conformal_calibrated(self.predictor):
            msg = "The model must first be calibrated before it can be represented."
            raise ValueError(msg)
        predictions = create_sample(predict_raw(self.predictor, *args, **kwargs))
        predictions = flatten_sample(predictions)
        weight_low, weight_high = self.predictor.non_conformity_score.weight(predictions)
        low_bound = predictions - weight_low * self.predictor.conformal_quantile
        upper_bound = predictions + weight_high * self.predictor.conformal_quantile
        return create_interval_conformal_set(low_bound, upper_bound)


@lazydispatch
def flatten_ensemble_quantile_sample[T](sample: Sample[T]) -> T:
    msg = f"Flattening not implemented for type {type(sample)}."
    raise NotImplementedError(msg)


@flatten_ensemble_quantile_sample.register(ArraySample)
def flatten_ensemble_quantile_sample_numpy(sample: ArraySample) -> np.ndarray:
    raw_array = sample.array
    if raw_array.ndim == 3:
        return sample.sample_mean()
    return raw_array


@representer.register(ConformalQuantileRegressionPredictor)
class ConformalQuantileRegressionRepresenter[**In, Out: ConformalSet](ConformalRepresenter[In, Out]):
    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> ConformalSet:
        """Predict for a conformal quantile regression predictor."""
        if not is_conformal_calibrated(self.predictor):
            msg = "The model must first be calibrated before it can be represented."
            raise ValueError(msg)
        predictions = create_sample(predict_raw(self.predictor, *args, **kwargs),sample_axis=0)
        weight_low, weight_high = self.predictor.non_conformity_score.weight(predictions)
        mean_predictions = flatten_ensemble_quantile_sample(predictions)
        low_bound = mean_predictions[:, 0] - self.predictor.conformal_quantile * weight_low
        upper_bound = mean_predictions[:, 1] + self.predictor.conformal_quantile * weight_high
        return create_interval_conformal_set(low_bound, upper_bound)
