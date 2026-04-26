"""Evidential classification implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE
from probly.method.evidential.classification import common
from probly.method.evidential.classification.common import EvidentialClassificationPredictor

evidential_classification = common.evidential_classification
register = common.register


## Torch
@common.evidential_classification_appender.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["EvidentialClassificationPredictor", "evidential_classification", "register"]
