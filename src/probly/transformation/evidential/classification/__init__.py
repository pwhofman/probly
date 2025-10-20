"""Evidential classification implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE
from probly.transformation.evidential.classification import common

evidential_classification = common.evidential_classification
register = common.register


## Torch
@common.evidential_classification_appender.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415
