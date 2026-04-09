"""Shared credal net implementation."""

from __future__ import annotations
import probly
from typing import Literal

from typing import TYPE_CHECKING

from lazy_dispatch import lazydispatch
from probly.method.method import predictor_transformation
from probly.predictor import ProbabilisticClassifier

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor


@lazydispatch
def credal_net_generator[T: Predictor](base: T) -> T:
    """Generate a credal net from a base model."""
    ...


def register(cls: LazyType, generator: Callable) -> None:
    """Register a class which can be used as a base for a credal net."""
    ...
def credal_net[T: Predictor](base: T, *, predictor_type: Literal['categorical_distribution_predictor', 'probabilistic_classifier'] | type[probly.predictor.CategoricalDistributionPredictor] | None = None) -> T:
    """Create a credal net predictor from a base predictor based on :cite:`wang2024credalnet`.

    Args:
        base: Predictor, The base model to be used for credal net.
        num_classes: int, The number of classes to predict.

    Returns:
        Predictor, The credal net predictor.
    """
    ...
