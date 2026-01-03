"""Conformal Prediction LAC score implementation."""

from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR

from .common import LACScore, accretive_completion, lac_score_func


# Lazy registration - these will only be imported when needed
@lac_score_func.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from probly.conformal_prediction.scores.lac import torch  # noqa: PLC0415, F401


@lac_score_func.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from probly.conformal_prediction.scores.lac import flax  # noqa: PLC0415, F401


__all__ = ["LACScore", "accretive_completion", "lac_score_func"]
