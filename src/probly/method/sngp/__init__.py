"""SNGP: Spectral-normalized Neural Gaussian Process implementation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE, TORCH_SAMPLE

from ._common import (
    SNGPPredictor,
    SNGPRepresenter,
    _collect_skipped_param_bearing_layer_classes,
    compute_categorical_sample_from_logits,
    register,
    reset_precision_matrix,
    sngp,
    sngp_traverser,
)


## Torch
@sngp_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@compute_categorical_sample_from_logits.delayed_register(TORCH_SAMPLE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@reset_precision_matrix.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@_collect_skipped_param_bearing_layer_classes.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "SNGPPredictor",
    "SNGPRepresenter",
    "compute_categorical_sample_from_logits",
    "register",
    "reset_precision_matrix",
    "sngp",
    "sngp_traverser",
]
