"""Bindings for torch-uncertainty predictors and probly representations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, overload, override

import torch
from torch_uncertainty.models.wrappers.batch_ensemble import BatchEnsemble
from torch_uncertainty.models.wrappers.checkpoint_collector import CheckpointCollector
from torch_uncertainty.models.wrappers.deep_ensembles import _DeepEnsembles, _RegDeepEnsembles
from torch_uncertainty.models.wrappers.mc_dropout import _MCDropout, _RegMCDropout
from torch_uncertainty.models.wrappers.stochastic import StochasticModel
from torch_uncertainty.models.wrappers.swag import SWAG
from torch_uncertainty.post_processing.calibration.bbq import BBQScaler
from torch_uncertainty.post_processing.calibration.dirichlet_scaler import DirichletScaler
from torch_uncertainty.post_processing.calibration.histogram_binning import HistogramBinningScaler
from torch_uncertainty.post_processing.calibration.isotonic_regression import IsotonicRegressionScaler
from torch_uncertainty.post_processing.calibration.matrix_scaler import MatrixScaler
from torch_uncertainty.post_processing.calibration.temperature_scaler import TemperatureScaler
from torch_uncertainty.post_processing.calibration.vector_scaler import VectorScaler
from torch_uncertainty.post_processing.conformal.aps import ConformalClsAPS
from torch_uncertainty.post_processing.conformal.raps import ConformalClsRAPS
from torch_uncertainty.post_processing.conformal.thr import ConformalClsTHR
from torch_uncertainty.post_processing.mc_batch_norm import MCBatchNorm

from probly.predictor import Predictor, predict, predict_raw
from probly.representation.conformal_set.torch import TorchOneHotConformalSet
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
    TorchLogitCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution, TorchGaussianDistributionSample
from probly.representation.sample.torch import TorchSample
from probly.representer import Representer, representer

type TorchUncertaintySampleKind = Literal["values", "logits", "probabilities", "gaussian"]

_BATCHED_WRAPPER_TYPES = (
    BatchEnsemble,
    CheckpointCollector,
    _DeepEnsembles,
    _RegDeepEnsembles,
    _MCDropout,
    _RegMCDropout,
    StochasticModel,
    SWAG,
    MCBatchNorm,
)

_CALIBRATION_TYPES = (
    BBQScaler,
    DirichletScaler,
    HistogramBinningScaler,
    IsotonicRegressionScaler,
    MatrixScaler,
    TemperatureScaler,
    VectorScaler,
)

_CONFORMAL_TYPES = (ConformalClsAPS, ConformalClsRAPS, ConformalClsTHR)


def _first_tensor_batch_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                msg = "Cannot infer batch size from a scalar torch.Tensor."
                raise ValueError(msg)
            return int(value.shape[0])
    msg = "torch-uncertainty integrations require at least one torch.Tensor input."
    raise ValueError(msg)


def _num_estimators(predictor: Any) -> int:  # noqa: ANN401
    if hasattr(predictor, "num_estimators"):
        return int(predictor.num_estimators)
    if hasattr(predictor, "num_samples"):
        return int(predictor.num_samples)
    msg = f"Cannot infer number of estimators for {type(predictor)}."
    raise ValueError(msg)


def _reshape_tensor_sample(tensor: torch.Tensor, num_estimators: int, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] != num_estimators * batch_size:
        msg = (
            f"Expected first output dimension to be num_estimators * batch_size "
            f"({num_estimators} * {batch_size}), got {tensor.shape[0]}."
        )
        raise ValueError(msg)
    return tensor.reshape(num_estimators, batch_size, *tensor.shape[1:])


@overload
def _reshape_batched_output(
    output: torch.Tensor, num_estimators: int, batch_size: int
) -> TorchSample[torch.Tensor]: ...


@overload
def _reshape_batched_output(
    output: Mapping[str, torch.Tensor], num_estimators: int, batch_size: int
) -> Mapping[str, TorchSample[torch.Tensor]]: ...


def _reshape_batched_output(
    output: torch.Tensor | Mapping[str, torch.Tensor], num_estimators: int, batch_size: int
) -> TorchSample[torch.Tensor] | Mapping[str, TorchSample[torch.Tensor]]:
    if isinstance(output, torch.Tensor):
        return TorchSample(_reshape_tensor_sample(output, num_estimators, batch_size), sample_dim=0)
    if isinstance(output, Mapping):
        return {
            key: TorchSample(_reshape_tensor_sample(value, num_estimators, batch_size), sample_dim=0)
            for key, value in output.items()
            if isinstance(value, torch.Tensor)
        }
    msg = f"Unsupported torch-uncertainty output type {type(output)}."
    raise TypeError(msg)


def _as_categorical_sample(
    sample: TorchSample[torch.Tensor], kind: Literal["logits", "probabilities"]
) -> TorchCategoricalDistributionSample:
    tensor = sample.tensor
    if kind == "logits":
        distribution = TorchLogitCategoricalDistribution(tensor)
    else:
        distribution = TorchProbabilityCategoricalDistribution(tensor)
    return TorchCategoricalDistributionSample(distribution, sample_dim=sample.sample_dim)


def _as_gaussian_sample(samples: Mapping[str, TorchSample[torch.Tensor]]) -> TorchGaussianDistributionSample:
    if "loc" in samples:
        mean = samples["loc"].tensor
    elif "mean" in samples:
        mean = samples["mean"].tensor
    else:
        msg = "Gaussian torch-uncertainty output requires a 'loc' or 'mean' tensor."
        raise ValueError(msg)

    if "var" in samples:
        var = samples["var"].tensor
    elif "variance" in samples:
        var = samples["variance"].tensor
    elif "scale" in samples:
        var = samples["scale"].tensor.square()
    else:
        msg = "Gaussian torch-uncertainty output requires a 'var', 'variance', or 'scale' tensor."
        raise ValueError(msg)

    return TorchGaussianDistributionSample(TorchGaussianDistribution(mean=mean, var=var), sample_dim=0)


@predict.register(_BATCHED_WRAPPER_TYPES)
def _predict_batched_torch_uncertainty[**In, Out](
    predictor: Predictor[In, Out], *args: In.args, **kwargs: In.kwargs
) -> TorchSample[torch.Tensor] | Mapping[str, TorchSample[torch.Tensor]]:
    batch_size = _first_tensor_batch_size(args, kwargs)
    raw = predict_raw(predictor, *args, **kwargs)
    return _reshape_batched_output(raw, _num_estimators(predictor), batch_size)


class TorchUncertaintySampleRepresenter[**In, Out](Representer[Any, In, Out, Any]):
    """Representer for torch-uncertainty wrappers with batched estimator outputs."""

    def __init__(self, predictor: Predictor[In, Out], kind: TorchUncertaintySampleKind = "values") -> None:
        super().__init__(predictor)
        self.kind = kind

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> Any:
        representation = predict(self.predictor, *args, **kwargs)
        if self.kind == "values":
            return representation
        if self.kind in ("logits", "probabilities"):
            if not isinstance(representation, TorchSample):
                msg = f"Expected tensor sample output for kind={self.kind!r}, got {type(representation)}."
                raise TypeError(msg)
            return _as_categorical_sample(representation, self.kind)
        if self.kind == "gaussian":
            if not isinstance(representation, Mapping):
                msg = f"Expected mapping output for kind='gaussian', got {type(representation)}."
                raise TypeError(msg)
            return _as_gaussian_sample(representation)
        msg = f"Unknown torch-uncertainty representation kind {self.kind!r}."
        raise ValueError(msg)


representer.register(_BATCHED_WRAPPER_TYPES, TorchUncertaintySampleRepresenter)


@predict.register(_CALIBRATION_TYPES)
def _predict_torch_uncertainty_calibrator[**In, Out](
    predictor: Predictor[In, Out], *args: In.args, **kwargs: In.kwargs
) -> Out:
    return predict_raw(predictor, *args, **kwargs)


class TorchUncertaintyCalibratedLogitRepresenter(Representer[Any, Any, Any, TorchLogitCategoricalDistribution]):
    """Representer for torch-uncertainty calibration scalers that return logits."""

    @override
    def represent(self, *args: Any, **kwargs: Any) -> TorchLogitCategoricalDistribution:
        return TorchLogitCategoricalDistribution(predict(self.predictor, *args, **kwargs))


representer.register(_CALIBRATION_TYPES, TorchUncertaintyCalibratedLogitRepresenter)


@predict.register(_CONFORMAL_TYPES)
def _predict_torch_uncertainty_conformal[**In, Out](
    predictor: Predictor[In, Out], *args: In.args, **kwargs: In.kwargs
) -> TorchOneHotConformalSet:
    return TorchOneHotConformalSet(predict_raw(predictor, *args, **kwargs) > 0)


class TorchUncertaintyConformalRepresenter(Representer[Any, Any, Any, Any]):
    """Representer for torch-uncertainty conformal classifiers."""

    @override
    def represent(self, *args: Any, **kwargs: Any) -> Any:
        return predict(self.predictor, *args, **kwargs)


representer.register(_CONFORMAL_TYPES, TorchUncertaintyConformalRepresenter)


__all__ = []
