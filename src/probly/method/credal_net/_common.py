"""Shared credal net implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.method.method import predictor_transformation
from probly.predictor import (
    LogitClassifier,
    Predictor,
    ProbabilisticClassifier,
    RepresentationPredictor,
    predict,
    predict_raw,
)
from probly.representation.credal_set import (
    ProbabilityIntervalsCredalSet,
    create_probability_intervals_from_lower_upper_array,
)
from probly.traverse_nn import nn_compose
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse_with_state


@runtime_checkable
class CredalNetPredictor[**In, Out: ProbabilityIntervalsCredalSet](RepresentationPredictor[In, Out], Protocol):
    """A predictor that predicts according to a credal interval net based on :cite:`wang2024credalnet`."""


REPLACED = GlobalVariable[bool]("REPLACED", default=False)

credal_net_traverser = flexdispatch_traverser[object](name="credal_net_traverser")


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier, LogitClassifier),
    preserve_predictor_type=False,
)  # ty:ignore[invalid-argument-type]
@CredalNetPredictor.register_factory
def credal_net[**In, Out: ProbabilityIntervalsCredalSet](base: Predictor[In, Out]) -> CredalNetPredictor[In, Out]:
    """Create a credal net predictor from a base classifier based on :cite:`wang2024credalnet`.

    Replaces every ``Conv2d``, ``BatchNorm2d``, ``BatchNorm1d``, and ``Linear``
    in the base network with its interval counterpart, and replaces the last
    ``Linear`` with the credal head ``IntLinear -> IntBatchNorm1d ->
    IntSoftmax``. Any trailing softmax in a ``ProbabilisticClassifier`` base
    is stripped.

    Args:
        base: Base predictor; must be a ``ProbabilisticClassifier`` or
            ``LogitClassifier``.

    Returns:
        The transformed credal net predictor.

    Raises:
        ValueError: If ``base`` contains no ``nn.Linear`` for the credal head
            to replace.
    """
    new_model, final_state = traverse_with_state(
        base, nn_compose(credal_net_traverser), init={REPLACED: False, TRAVERSE_REVERSED: True}
    )
    if not final_state[REPLACED]:
        msg = (
            "credal_net could not place the interval head: the base predictor has no Linear "
            "layer for the traverser to replace. Provide a ProbabilisticClassifier or "
            "LogitClassifier that ends in a linear layer."
        )
        raise ValueError(msg)
    return new_model


@predict.register(CredalNetPredictor)
def _[**In](
    predictor: CredalNetPredictor[In, ProbabilityIntervalsCredalSet], *args: In.args, **kwargs: In.kwargs
) -> ProbabilityIntervalsCredalSet:
    """Wrap the credal-net's packed interval output as a ``ProbabilityIntervalsCredalSet``."""
    return create_probability_intervals_from_lower_upper_array(predict_raw(predictor, *args, **kwargs))
