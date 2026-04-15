"""This module contains common code for conformal representers."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, override

from lazy_dispatch.singledispatch import lazydispatch
from probly.calibrator._common import ConformalCalibrator
from probly.conformal_scores._common import (
    AbsoluteErrorScore,
    ClassificationNonConformityScore,
    CQRrScore,
    CQRScore,
    UACQRScore,
)
from probly.predictor import ProbabilisticClassifier
from probly.predictor._common import (
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

if TYPE_CHECKING:
    import numpy as np

    from probly.calibrator._common import ConformalCalibrator
    from probly.representation.sample._common import Sample

if TYPE_CHECKING:
    import numpy as np

    from probly.calibrator._common import ConformalCalibrator
    from probly.representation.sample._common import Sample


def is_conformal_calibrated(calibrator: ConformalCalibrator[Any, Any]) -> bool:
    """Check if a conformal calibrator has been calibrated."""
    conformal_quantile = getattr(calibrator, "conformal_quantile", None)
    non_conformity_score = getattr(calibrator, "non_conformity_score", None)
    return conformal_quantile is not None and non_conformity_score is not None


def dispatch_on_score(predictor: ConformalCalibrator[Any, Any], *args: Any, **kwargs: Any) -> type:
    """Dispatch on the type of the predictor."""
    return predictor.non_conformity_score


@lazydispatch(dispatch_on=dispatch_on_score)
def predict_set(predictor: ConformalCalibrator[Any, Any], predictions: Sample[Any]) -> ConformalSet:
    """Predict a conformal set for the given predictions."""
    msg = f"Prediction not implemented for type {type(predictor)}."
    raise NotImplementedError(msg)


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
        set_prediction = predict_set(copy.deepcopy(self.predictor), *args, **kwargs)
        return set_prediction


@predict_set.register(ClassificationNonConformityScore)
def predict_set_classification(predictor: ConformalCalibrator[Any, Any], *args, **kwargs: Any) -> OneHotConformalSet:
    """Standard conformal classification prediction."""
    # We register the predictor as a probabilistic classifier to obtain the raw predictions for the conformal set construction.
    ProbabilisticClassifier.register_instance(predictor)
    predictions = predict_raw(predictor, *args, **kwargs)
    non_conformity_score = predictor.non_conformity_score(predictions)
    set_prediction = non_conformity_score <= predictor.conformal_quantile
    return create_onehot_conformal_set(set_prediction)


@predict_set.register(AbsoluteErrorScore)
def predict_set_regression(predictor: ConformalCalibrator[Any, Any], *args, **kwargs: Any) -> ConformalSet:
    """Standard conformal regression prediction."""
    predictions = predict_raw(predictor, *args, **kwargs)
    low_bound = predictions - predictor.conformal_quantile
    upper_bound = predictions + predictor.conformal_quantile
    return create_interval_conformal_set(low_bound, upper_bound)


@predict_set.register(CQRScore)
def predict_set_cqr(predictor: ConformalCalibrator[Any, Any], *args, **kwargs: Any) -> ConformalSet:
    """Conformalized quantile regression prediction."""
    predictions = predict_raw(predictor, *args, **kwargs)
    low_bound = predictions[:, 0] - predictor.conformal_quantile
    upper_bound = predictions[:, 1] + predictor.conformal_quantile
    return create_interval_conformal_set(low_bound, upper_bound)


@predict_set.register(CQRrScore)
def predict_set_cqr_r(predictor: ConformalCalibrator[Any, Any], *args, **kwargs: Any) -> ConformalSet:
    """Conformalized quantile regression with residuals prediction."""
    predictions = predict_raw(predictor, *args, **kwargs)
    width = predictions[:, 1] - predictions[:, 0]
    low_bound = predictions[:, 0] - predictor.conformal_quantile * width
    upper_bound = predictions[:, 1] + predictor.conformal_quantile * width
    return create_interval_conformal_set(low_bound, upper_bound)


@predict_set.register(UACQRScore)
def predict_set_uacqr(predictor: ConformalCalibrator[Any, Any], *args, **kwargs: Any) -> ConformalSet:
    """Uncertainty-aware conformalized quantile regression prediction."""
    predictions = predict_raw(predictor, *args, **kwargs)
    mean_predictions = predictions.sample_mean()
    standard_deviation = predictions.sample_std()
    weight_low, weight_high = standard_deviation[:, 0], standard_deviation[:, 1]
    low_bound = mean_predictions[:, 0] - predictor.conformal_quantile * weight_low
    upper_bound = mean_predictions[:, 1] + predictor.conformal_quantile * weight_high
    return create_interval_conformal_set(low_bound, upper_bound)
