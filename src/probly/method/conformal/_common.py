"""Conformal prediction wrappers and factory functions."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Protocol, Self, cast, runtime_checkable

from flextype import flexdispatch
from probly.calibrator._common import Calibrator
from probly.conformal_scores import (
    APSScore,
    RAPSScore,
    SAPSScore,
    absolute_error_score,
    cqr_r_score,
    cqr_score,
    lac_score,
    uacqr_score,
)
from probly.method.method import predictor_transformation
from probly.predictor import (
    IterablePredictor,
    LogitClassifier,
    Predictor,
    ProbabilisticClassifier,
    RepresentationPredictor,
    predict,
)
from probly.representation.conformal_set._common import (
    ConformalSet,
    IntervalConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from probly.representation.sample import create_sample
from probly.representation.sample._common import Sample
from probly.utils.quantile._common import calculate_quantile

if TYPE_CHECKING:
    from probly.conformal_scores._common import NonConformityScore


@runtime_checkable  # ty:ignore[conflicting-metaclass]
class ConformalSetPredictor[**In, T, Out: ConformalSet](
    RepresentationPredictor[In, Out],
    Calibrator[In, T],
    Protocol,
):
    """Predictor wrapper returning conformal sets."""

    predictor: Predictor[In, Any]
    non_conformity_score: NonConformityScore[Any, Any] | None
    conformal_quantile: float | None

    def calibrate(self, alpha: float, y_calib: Out, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate the predictor on calibration data."""
        ...


@runtime_checkable
class ClassificationConformalSetPredictor[**In, T](ConformalSetPredictor[In, T, OneHotConformalSet], Protocol):
    """Conformal predictor for classification sets."""


@runtime_checkable
class RegressionConformalSetPredictor[**In, T](ConformalSetPredictor[In, T, IntervalConformalSet], Protocol):
    """Conformal predictor for interval regression sets."""


@runtime_checkable
class CQRConformalSetPredictor[**In, T](RegressionConformalSetPredictor[In, T], Protocol):
    """Conformal predictor for CQR sets."""


@runtime_checkable
class CQRrConformalSetPredictor[**In, T](RegressionConformalSetPredictor[In, T], Protocol):
    """Conformal predictor for CQR-r sets."""


@runtime_checkable
class UACQRConformalSetPredictor[**In, T](RegressionConformalSetPredictor[In, T], Protocol):
    """Conformal predictor for UACQR sets."""


@runtime_checkable
class LACConformalSetPredictor[**In, T](ClassificationConformalSetPredictor[In, T], Protocol):
    """Conformal predictor specialized for LAC."""


@runtime_checkable
class APSConformalSetPredictor[**In, T](ClassificationConformalSetPredictor[In, T], Protocol):
    """Conformal predictor specialized for APS."""


@runtime_checkable
class SAPSConformalSetPredictor[**In, T](ClassificationConformalSetPredictor[In, T], Protocol):
    """Conformal predictor specialized for SAPS."""


@runtime_checkable
class RAPSConformalSetPredictor[**In, T](ClassificationConformalSetPredictor[In, T], Protocol):
    """Conformal predictor specialized for RAPS."""


@runtime_checkable
class AbsoluteErrorConformalSetPredictor[**In, T](RegressionConformalSetPredictor[In, T], Protocol):
    """Conformal predictor specialized for absolute error scores."""


class _ConformalPredictorBase[**In, Out](ABC):
    """Backend-agnostic conformal predictor behavior."""

    predictor: Predictor[In, Out]
    non_conformity_score: NonConformityScore[Out, Out] | None
    conformal_quantile: float | None

    def __init__(
        self,
        predictor: Predictor[In, Out],
        non_conformity_score: NonConformityScore[Out, Out],
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.non_conformity_score = non_conformity_score
        self.conformal_quantile = None

    def _require_score(self) -> NonConformityScore[Out, Out]:
        if self.non_conformity_score is None:
            msg = "No non_conformity_score configured for this conformal predictor."
            raise ValueError(msg)
        return self.non_conformity_score

    def _require_calibrated(self) -> tuple[float, NonConformityScore[Out, Out]]:
        quantile = self.conformal_quantile
        if quantile is None:
            msg = "Conformal predictor is not calibrated. Please call calibrate() before prediction."
            raise ValueError(msg)
        score = self._require_score()
        return quantile, score

    def calibrate(self, alpha: float, y_calib: Out, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate the predictor using calibration data."""
        score = self._require_score()
        prediction = predict(self.predictor, *calib_args, **calib_kwargs)
        scores = score(prediction, y_calib)
        self.conformal_quantile = calculate_quantile(scores, alpha)
        return self


@flexdispatch
def calibrated_state(_: object) -> tuple[float, NonConformityScore[Any, Any]]:
    msg = "Predictor is not a conformal predictor or is not calibrated."
    raise ValueError(msg)


@calibrated_state.register(_ConformalPredictorBase)
def _[**In, Out](
    predictor: _ConformalPredictorBase[In, Out],
) -> tuple[float, NonConformityScore[Out, Out]]:
    return predictor._require_calibrated()  # noqa: SLF001


@predict.register(ClassificationConformalSetPredictor)
def predict_classification_conformal_set[**In, T](
    predictor: ClassificationConformalSetPredictor[In, T],
    *args: In.args,
    **kwargs: In.kwargs,
) -> OneHotConformalSet:
    """Predict a classification conformal set."""
    quantile, score = calibrated_state(predictor)
    prediction = predict(cast("Any", predictor).predictor, *args, **kwargs)
    scores = score(prediction)
    return create_onehot_conformal_set(scores <= quantile)


@predict.register(AbsoluteErrorConformalSetPredictor)
def predict_absolute_error_conformal_set[**In, T](
    predictor: AbsoluteErrorConformalSetPredictor[In, T],
    *args: In.args,
    **kwargs: In.kwargs,
) -> IntervalConformalSet:
    """Predict an absolute-error conformal interval."""
    quantile, _score = calibrated_state(predictor)
    prediction = predict(cast("Any", predictor).predictor, *args, **kwargs)
    lower = prediction - quantile
    upper = prediction + quantile
    return create_interval_conformal_set(lower, upper)


def _quantile_prediction_mean(predictor: object, prediction: object) -> object:
    if isinstance(predictor, IterablePredictor) or isinstance(prediction, Sample):
        return create_sample(prediction, sample_axis=0).sample_mean()
    return prediction


@predict.register(CQRConformalSetPredictor)
def predict_cqr_conformal_set[**In, T](
    predictor: CQRConformalSetPredictor[In, T],
    *args: In.args,
    **kwargs: In.kwargs,
) -> IntervalConformalSet:
    """Predict a CQR conformal interval."""
    quantile, _score = calibrated_state(predictor)
    prediction = predict(cast("Any", predictor).predictor, *args, **kwargs)
    prediction = cast("Any", _quantile_prediction_mean(cast("Any", predictor).predictor, prediction))
    lower = prediction[..., 0] - quantile
    upper = prediction[..., 1] + quantile
    return create_interval_conformal_set(lower, upper)


@predict.register(CQRrConformalSetPredictor)
def predict_cqr_r_conformal_set[**In, T](
    predictor: CQRrConformalSetPredictor[In, T],
    *args: In.args,
    **kwargs: In.kwargs,
) -> IntervalConformalSet:
    """Predict a CQR-r conformal interval."""
    quantile, _score = calibrated_state(predictor)
    prediction = predict(cast("Any", predictor).predictor, *args, **kwargs)
    prediction = cast("Any", _quantile_prediction_mean(cast("Any", predictor).predictor, prediction))
    width = prediction[..., 1] - prediction[..., 0]
    lower = prediction[..., 0] - quantile * width
    upper = prediction[..., 1] + quantile * width
    return create_interval_conformal_set(lower, upper)


@predict.register(UACQRConformalSetPredictor)
def predict_uacqr_conformal_set[**In, T](
    predictor: UACQRConformalSetPredictor[In, T],
    *args: In.args,
    **kwargs: In.kwargs,
) -> IntervalConformalSet:
    """Predict a UACQR conformal interval."""
    quantile, _score = calibrated_state(predictor)
    prediction = predict(cast("Any", predictor).predictor, *args, **kwargs)
    if isinstance(cast("Any", predictor).predictor, IterablePredictor) or isinstance(prediction, Sample):
        sample = create_sample(prediction, sample_axis=0)
        mean_prediction = sample.sample_mean()
        std_prediction = sample.sample_std()
        lower = mean_prediction[..., 0] - quantile * std_prediction[..., 0]
        upper = mean_prediction[..., 1] + quantile * std_prediction[..., 1]
    else:
        prediction = cast("Any", prediction)
        lower = prediction[..., 0]
        upper = prediction[..., 1]

    return create_interval_conformal_set(lower, upper)


@flexdispatch
def conformal_generator[**In, T, Out](
    base: Predictor[In, Out],
    non_conformity_score: NonConformityScore[Out, T],
) -> ConformalSetPredictor[In, T, ConformalSet]:
    """Generate a backend-specific conformal set predictor wrapper."""
    msg = f"No conformal generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier, LogitClassifier), preserve_predictor_type=False
)
@LACConformalSetPredictor.register_factory
def conformal_lac[**In, T, Out](base: Predictor[In, Out]) -> LACConformalSetPredictor[In, T]:
    """Create a LAC conformal predictor wrapper."""
    return conformal_generator(base, lac_score)  # ty:ignore[invalid-argument-type]


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier, LogitClassifier), preserve_predictor_type=False
)
@APSConformalSetPredictor.register_factory
def conformal_aps[**In, T, Out](
    base: Predictor[In, Out],
    randomized: bool = True,
) -> APSConformalSetPredictor[In, T]:
    """Create an APS conformal predictor wrapper."""
    return conformal_generator(base, APSScore(randomized=randomized))


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier, LogitClassifier), preserve_predictor_type=False
)
@SAPSConformalSetPredictor.register_factory
def conformal_saps[**In, T, Out](
    base: Predictor[In, Out],
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> SAPSConformalSetPredictor[In, T]:
    """Create a SAPS conformal predictor wrapper."""
    return conformal_generator(
        base,
        SAPSScore(randomized=randomized, lambda_val=lambda_val),
    )


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier, LogitClassifier), preserve_predictor_type=False
)
@RAPSConformalSetPredictor.register_factory
def conformal_raps[**In, T, Out](
    base: Predictor[In, Out],
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> RAPSConformalSetPredictor[In, T]:
    """Create a RAPS conformal predictor wrapper."""
    return conformal_generator(
        base,
        RAPSScore(
            randomized=randomized,
            lambda_reg=lambda_reg,
            k_reg=k_reg,
            epsilon=epsilon,
        ),
    )


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@AbsoluteErrorConformalSetPredictor.register_factory
def conformal_absolute_error[**In, T, Out](base: Predictor[In, Out]) -> AbsoluteErrorConformalSetPredictor[In, T]:
    """Create an absolute-error conformal predictor wrapper."""
    return conformal_generator(base, absolute_error_score)  # ty:ignore[invalid-argument-type]


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@CQRConformalSetPredictor.register_factory
def conformal_cqr[**In, T, Out](base: Predictor[In, Out]) -> CQRConformalSetPredictor[In, T]:
    """Create a CQR conformal predictor wrapper."""
    return conformal_generator(base, cqr_score)  # ty:ignore[invalid-argument-type]


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@CQRrConformalSetPredictor.register_factory
def conformal_cqr_r[**In, T, Out](base: Predictor[In, Out]) -> CQRrConformalSetPredictor[In, T]:
    """Create a CQR-r conformal predictor wrapper."""
    return conformal_generator(base, cqr_r_score)  # ty:ignore[invalid-argument-type]


@predictor_transformation(permitted_predictor_types=(IterablePredictor,), preserve_predictor_type=False)
@UACQRConformalSetPredictor.register_factory
def conformal_uacqr[**In, T, Out](base: Predictor[In, Out]) -> UACQRConformalSetPredictor[In, T]:
    """Create a UACQR conformal predictor wrapper."""
    return conformal_generator(base, uacqr_score)  # ty:ignore[invalid-argument-type]


__all__ = [
    "APSConformalSetPredictor",
    "AbsoluteErrorConformalSetPredictor",
    "CQRConformalSetPredictor",
    "CQRrConformalSetPredictor",
    "ClassificationConformalSetPredictor",
    "ConformalSetPredictor",
    "LACConformalSetPredictor",
    "RAPSConformalSetPredictor",
    "SAPSConformalSetPredictor",
    "UACQRConformalSetPredictor",
    "_ConformalPredictorBase",
    "conformal_absolute_error",
    "conformal_aps",
    "conformal_cqr",
    "conformal_cqr_r",
    "conformal_generator",
    "conformal_lac",
    "conformal_raps",
    "conformal_saps",
    "conformal_uacqr",
]
