"""RBF centroid-head transformation for classification models."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import RBFCentroidHeadPredictor, rbf_centroid_head, rbf_centroid_head_generator


@rbf_centroid_head_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "RBFCentroidHeadPredictor",
    "rbf_centroid_head",
    "rbf_centroid_head_generator",
]
