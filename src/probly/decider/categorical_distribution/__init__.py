"""Deciders for reducing representations to categorical distributions."""

from probly.lazy_types import TORCH_VALUE

from . import numpy as numpy
from ._common import categorical_from_maximin, categorical_from_mean, mean_field_categorical


@categorical_from_mean.delayed_register(TORCH_VALUE)
@categorical_from_maximin.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["categorical_from_maximin", "categorical_from_mean", "mean_field_categorical"]
