"""Dropout ensemble implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, SKLEARN_MODULE, TORCH_MODULE

from ._common import ConformalClassificationCalibrator, conformal_generator, conformalize_classifier


@conformal_generator.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


@conformal_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@conformal_generator.delayed_register(SKLEARN_MODULE)
def _(_: type) -> None:
    from . import sklearn as sklearn  # noqa: PLC0415


__all__ = ["ConformalClassificationCalibrator", "conformalize_classifier"]
