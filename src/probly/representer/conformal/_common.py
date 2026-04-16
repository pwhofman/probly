"""This module contains common code for conformal representers."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, override, runtime_checkable

from lazy_dispatch.singledispatch import lazydispatch
from probly.calibrator._common import ConformalCalibrator
from probly.conformal_scores._common import (
    AbsoluteErrorScore,
    ClassificationNonConformityScore,
    CQRrScore,
    CQRScore,
    UACQRScore,
)
from probly.method.conformal._common import ConformalClassificationCalibrator, ensure_distribution_2d
from probly.predictor._common import (
    IterablePredictor,
    RandomPredictor,
    RepresentationPredictor,
    predict_raw,
)
from probly.representation.conformal_set._common import (
    ConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from probly.representation.sample import create_sample
from probly.representation.sample.array import ArraySample
from probly.representer import representer
from probly.representer._representer import Representer

if TYPE_CHECKING:
    import numpy as np

    from probly.representation.sample._common import Sample

if TYPE_CHECKING:
    import numpy as np

    from probly.calibrator._common import ConformalCalibrator
    from probly.representation.sample._common import Sample


@runtime_checkable  # ty: ignore[conflicting-metaclass]
class ConformalIterableCalibrator[**In, Out](
    ConformalCalibrator[In, Out],
    IterablePredictor[In, Out],
    Protocol,
):
    """Intersection protocol for conformal calibrators that are iterable predictors."""

    _running_instancehook: ClassVar[ContextVar[object]] = ContextVar(
        "ConformalIterableCalibrator._running_instancehook", default=NotImplementedError
    )

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        if cls._running_instancehook.get() is instance:
            return NotImplemented
        try:
            tok = cls._running_instancehook.set(instance)
            if isinstance(instance, ConformalClassificationCalibrator):
                return False
            if isinstance(instance, ConformalCalibrator) and isinstance(instance, IterablePredictor):
                return True

        finally:
            cls._running_instancehook.reset(tok)
        return NotImplemented


@runtime_checkable  # ty: ignore[conflicting-metaclass]
class ConformalRandomCalibrator[In, Out](
    ConformalCalibrator[In, Out],  # ty: ignore[invalid-type-arguments]
    RandomPredictor[In, Out],  # ty: ignore[invalid-type-arguments]
    Protocol,
):
    """Intersection protocol for conformal calibrators that are random predictors."""

    _running_instancehook: ClassVar[ContextVar[object]] = ContextVar(
        "ConformalRandomCalibrator._running_instancehook", default=NotImplementedError
    )

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        if cls._running_instancehook.get() is instance:
            return NotImplemented
        try:
            tok = cls._running_instancehook.set(instance)
            if isinstance(instance, IterablePredictor):
                return False
            if isinstance(instance, RepresentationPredictor):
                return False
            if isinstance(instance, ConformalCalibrator) and isinstance(instance, RandomPredictor):
                return True
        finally:
            cls._running_instancehook.reset(tok)
        return NotImplemented


def is_conformal_calibrated(calibrator: ConformalCalibrator[Any, Any]) -> bool:
    """Check if a conformal calibrator has been calibrated."""
    conformal_quantile = getattr(calibrator, "conformal_quantile", None)
    non_conformity_score = getattr(calibrator, "non_conformity_score", None)
    return conformal_quantile is not None and non_conformity_score is not None


def dispatch_on_score(predictor: ConformalCalibrator[Any, Any], *_args: object, **_kwargs: object) -> object:
    """Dispatch on the type of the predictor."""
    return predictor.non_conformity_score


@lazydispatch(dispatch_on=dispatch_on_score)
def predict_set(predictor: ConformalCalibrator[Any, Any], *args: object, **kwargs: object) -> ConformalSet:
    """Predict a conformal set for the given predictions."""
    _ = (args, kwargs)
    msg = f"Prediction not implemented for type {type(predictor)}."
    raise NotImplementedError(msg)


@representer.register(ConformalRandomCalibrator)
@representer.register(ConformalIterableCalibrator)
@representer.register(ConformalClassificationCalibrator)
@representer.register(ConformalCalibrator)
class ConformalRepresenter[**In, Out: ConformalSet](Representer[Any, In, Out, ConformalSet]):
    predictor: ConformalCalibrator[In, Out]

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> ConformalSet:
        """Represent the output of a conformal predictor as a conformal set."""
        if not is_conformal_calibrated(self.predictor):
            msg = "The model must first be calibrated before it can be represented."
            raise ValueError(msg)
        # We dispatch
        set_prediction = predict_set(self.predictor, *args, **kwargs)
        return set_prediction


@lazydispatch
def ensure1d(prediction: Sample[Any]) -> np.ndarray:
    """Ensure that the prediction has a single data dimension."""
    msg = f"Cannot ensure 1D for prediction of type {type(prediction)}."
    raise NotImplementedError(msg)


@lazydispatch
def ensure2d[In](prediction: Sample[In]) -> In:
    """Ensure that the prediction has a sample dimension and a class dimension."""
    msg = f"Cannot ensure 2D for prediction of type {type(prediction)}."
    raise NotImplementedError(msg)


@ensure1d.register(ArraySample)
def _(prediction: ArraySample[Any]) -> np.ndarray:
    """Ensure that the prediction has a single data dimension."""
    data_a = prediction.array
    if data_a.ndim == 1:
        return data_a
    if data_a.ndim == 2:
        return data_a.mean(axis=0)
    msg = "Predictions with more than 2 dimensions are not supported for conformal regression."
    raise ValueError(msg)


@ensure2d.register(ArraySample)
def _(prediction: ArraySample[Any]) -> np.ndarray:
    """Ensure that the prediction has a sample dimension and a data dimension."""
    data_a = prediction.array
    if data_a.ndim == 1:
        return data_a.reshape(-1, 1)
    if data_a.ndim == 2:
        return data_a
    if data_a.ndim == 3:
        return data_a.mean(axis=0)
    msg = "Predictions with more than 3 dimensions are not supported for conformal quantile regression."
    raise ValueError(msg)


@predict_set.register(ClassificationNonConformityScore)
def predict_set_classification(
    predictor: ConformalCalibrator[Any, Any], *args: object, **kwargs: object
) -> OneHotConformalSet:
    """Standard conformal classification prediction."""
    # Use raw predictions for conformal set construction.
    predictions = create_sample(predict_raw(predictor, *args, **kwargs), sample_axis=0)
    probabilities = ensure_distribution_2d(predictions)
    calibrated_predictor = cast("Any", predictor)
    non_conformity_score = calibrated_predictor.non_conformity_score(probabilities)
    set_prediction = non_conformity_score <= calibrated_predictor.conformal_quantile
    return create_onehot_conformal_set(set_prediction)


@predict_set.register(AbsoluteErrorScore)
def predict_set_regression(predictor: ConformalCalibrator[Any, Any], *args: object, **kwargs: object) -> ConformalSet:
    """Standard conformal regression prediction."""
    predictions = create_sample(predict_raw(predictor, *args, **kwargs), sample_axis=0)
    predictions = ensure1d(predictions)
    calibrated_predictor = cast("Any", predictor)
    low_bound = predictions - calibrated_predictor.conformal_quantile
    upper_bound = predictions + calibrated_predictor.conformal_quantile
    return create_interval_conformal_set(low_bound, upper_bound)


@predict_set.register(CQRScore)
def predict_set_cqr(predictor: ConformalCalibrator[Any, Any], *args: object, **kwargs: object) -> ConformalSet:
    """Conformalized quantile regression prediction."""
    predictions = create_sample(predict_raw(predictor, *args, **kwargs), sample_axis=0)
    predictions = ensure2d(predictions)
    calibrated_predictor = cast("Any", predictor)
    low_bound = predictions[:, 0] - calibrated_predictor.conformal_quantile
    upper_bound = predictions[:, 1] + calibrated_predictor.conformal_quantile
    return create_interval_conformal_set(low_bound, upper_bound)


@predict_set.register(CQRrScore)
def predict_set_cqr_r(predictor: ConformalCalibrator[Any, Any], *args: object, **kwargs: object) -> ConformalSet:
    """Conformalized quantile regression with residuals prediction."""
    predictions = create_sample(predict_raw(predictor, *args, **kwargs), sample_axis=0)
    predictions = ensure2d(predictions)
    calibrated_predictor = cast("Any", predictor)
    width = predictions[:, 1] - predictions[:, 0]
    low_bound = predictions[:, 0] - calibrated_predictor.conformal_quantile * width
    upper_bound = predictions[:, 1] + calibrated_predictor.conformal_quantile * width
    return create_interval_conformal_set(low_bound, upper_bound)


@predict_set.register(UACQRScore)
def predict_set_uacqr(predictor: ConformalCalibrator[Any, Any], *args: object, **kwargs: object) -> ConformalSet:
    """Uncertainty-aware conformalized quantile regression prediction."""
    predictions = create_sample(predict_raw(predictor, *args, **kwargs), sample_axis=0)
    mean_predictions = predictions.sample_mean()
    standard_deviation = predictions.sample_std()
    calibrated_predictor = cast("Any", predictor)
    weight_low, weight_high = standard_deviation[:, 0], standard_deviation[:, 1]
    low_bound = mean_predictions[:, 0] - calibrated_predictor.conformal_quantile * weight_low
    upper_bound = mean_predictions[:, 1] + calibrated_predictor.conformal_quantile * weight_high
    return create_interval_conformal_set(low_bound, upper_bound)
