"""Optional integrations with external uncertainty libraries."""

from __future__ import annotations

from probly.calibrator import calibrate
from probly.predictor import predict
from probly.representer import representer

_TU_BATCHED_WRAPPERS = (
    "torch_uncertainty.models.wrappers.batch_ensemble.BatchEnsemble",
    "torch_uncertainty.models.wrappers.checkpoint_collector.CheckpointCollector",
    "torch_uncertainty.models.wrappers.deep_ensembles._DeepEnsembles",
    "torch_uncertainty.models.wrappers.deep_ensembles._RegDeepEnsembles",
    "torch_uncertainty.models.wrappers.mc_dropout._MCDropout",
    "torch_uncertainty.models.wrappers.mc_dropout._RegMCDropout",
    "torch_uncertainty.models.wrappers.stochastic.StochasticModel",
    "torch_uncertainty.models.wrappers.swag.SWAG",
    "torch_uncertainty.post_processing.mc_batch_norm.MCBatchNorm",
)

_TU_PREDICTION_POSTPROCESSORS = (
    "torch_uncertainty.post_processing.calibration.bbq.BBQScaler",
    "torch_uncertainty.post_processing.calibration.dirichlet_scaler.DirichletScaler",
    "torch_uncertainty.post_processing.calibration.histogram_binning.HistogramBinningScaler",
    "torch_uncertainty.post_processing.calibration.isotonic_regression.IsotonicRegressionScaler",
    "torch_uncertainty.post_processing.calibration.matrix_scaler.MatrixScaler",
    "torch_uncertainty.post_processing.calibration.temperature_scaler.TemperatureScaler",
    "torch_uncertainty.post_processing.calibration.vector_scaler.VectorScaler",
    "torch_uncertainty.post_processing.conformal.aps.ConformalClsAPS",
    "torch_uncertainty.post_processing.conformal.raps.ConformalClsRAPS",
    "torch_uncertainty.post_processing.conformal.thr.ConformalClsTHR",
)


@predict.delayed_register(_TU_BATCHED_WRAPPERS + _TU_PREDICTION_POSTPROCESSORS)
@representer.delayed_register(_TU_BATCHED_WRAPPERS + _TU_PREDICTION_POSTPROCESSORS)
@calibrate.delayed_register(_TU_PREDICTION_POSTPROCESSORS)
def _(_: type[object]) -> None:
    from . import torch_uncertainty as torch_uncertainty  # noqa: PLC0415


__all__ = []
