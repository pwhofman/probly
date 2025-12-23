"""Init for LAC scores."""

from probly.lazy_types import TORCH_TENSOR

from .common import lac_score_func


@lac_score_func.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch  # noqa: PLC0415
