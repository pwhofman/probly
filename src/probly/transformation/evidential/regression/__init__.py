"""Evidential regression implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE
from probly.transformation.evidential.regression import common

evidential_regression = common.evidential_regression
register = common.register


## Torch
@common.evidential_regression_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415
