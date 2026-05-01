"""Shared class-bias ensemble transformation implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from probly.predictor import LogitClassifier
from probly.transformation.ensemble import EnsemblePredictor, register_ensemble_members
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import nn_compose, reset_traverser
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from probly.predictor import Predictor


class ClassBiasEnsemblePredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor whose ensemble members receive class-specific bias initialization."""


INITIALIZED = GlobalVariable[bool]("INITIALIZED", default=False)
RESET_PARAMS = GlobalVariable[bool]("RESET_PARAMS", default=True)
BIAS_CLS = GlobalVariable[int]("BIAS_CLS", default=0)
TOBIAS_VALUE = GlobalVariable[int]("TOBIAS_VALUE", default=100)

class_bias_ensemble_traverser = flexdispatch_traverser[object](name="class_bias_ensemble_traverser")


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), post_transform=register_ensemble_members)
@ClassBiasEnsemblePredictor.register_factory(autocast_builtins=True)
def class_bias_ensemble[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True, tobias_value: int = 100
) -> ClassBiasEnsemblePredictor[In, Out]:
    """Create an ensemble with class-specific final-layer bias initialization.

    Args:
        base: The base model to be used for the class-bias ensemble.
        num_members: The number of members in the class-bias ensemble.
        reset_params: Whether to reset the parameters of each member.
        tobias_value: The value to use for the class bias initialization.

    Returns:
        The class-bias ensemble predictor.
    """
    if reset_params:
        traverser = nn_compose(reset_traverser, class_bias_ensemble_traverser)
    else:
        traverser = nn_compose(class_bias_ensemble_traverser)
    members = [
        traverse(
            base,
            traverser,
            init={
                BIAS_CLS: i,
                TOBIAS_VALUE: tobias_value,
                INITIALIZED: False,
                RESET_PARAMS: reset_params,
                TRAVERSE_REVERSED: True,
            },
        )
        for i in range(num_members)
    ]
    return members  # ty:ignore[invalid-return-type]
