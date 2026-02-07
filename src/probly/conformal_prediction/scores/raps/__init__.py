"""Conformal Prediction RAPS scores implementation."""

from probly.conformal_prediction.scores.raps.common import RAPSScore, raps_score_func
from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR


# Lazy registration : these will only be imported when needed
@raps_score_func.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from probly.conformal_prediction.scores.raps import torch  # noqa: PLC0415,F401


@raps_score_func.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from probly.conformal_prediction.scores.raps import flax  # noqa: PLC0415,F401


__all__ = ["RAPSScore", "raps_score_func"]
