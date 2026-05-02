"""Shared interval classifier transformation implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import nn_compose
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse_with_state


@runtime_checkable
class IntervalClassifierPredictor[**In, Out: ProbabilityIntervalsCredalSet](RepresentationPredictor[In, Out], Protocol):
    """A predictor that returns packed lower and upper probability intervals."""


REPLACED = GlobalVariable[bool]("REPLACED", default=False)
USE_BASE_WEIGHTS = GlobalVariable[bool]("USE_BASE_WEIGHTS", default=False)

interval_classifier_traverser = flexdispatch_traverser[object](name="interval_classifier_traverser")


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier, LogitClassifier),
    preserve_predictor_type=False,
)  # ty:ignore[invalid-argument-type]
@IntervalClassifierPredictor.register_factory
def interval_classifier[**In, Out: ProbabilityIntervalsCredalSet](
    base: Predictor[In, Out],
    use_base_weights: bool = USE_BASE_WEIGHTS.default,
) -> IntervalClassifierPredictor[In, Out]:
    """Create an interval classifier from a base classifier.

    Replaces every ``Conv2d``, ``BatchNorm2d``, ``BatchNorm1d``, and ``Linear``
    in the base network with its interval counterpart, and replaces the last
    ``Linear`` with the credal head ``IntLinear -> IntBatchNorm1d ->
    IntSoftmax``. Any trailing softmax in a ``ProbabilisticClassifier`` base
    is stripped.

    Args:
        base: Base predictor; must be a ``ProbabilisticClassifier`` or
            ``LogitClassifier``.
        use_base_weights: If True, copy each replaced layer's weights, biases
            and (for BatchNorm) running statistics into the new layer's
            ``center_*`` slots. The radius parameters keep their fresh
            initialization. If False, every interval layer starts from
            scratch (matching how methods like dropout behave by default).

    Returns:
        The transformed interval classifier predictor.

    Raises:
        ValueError: If ``base`` contains no ``nn.Linear`` for the credal head
            to replace.
    """
    new_model, final_state = traverse_with_state(
        base,
        nn_compose(interval_classifier_traverser),
        init={REPLACED: False, USE_BASE_WEIGHTS: use_base_weights, TRAVERSE_REVERSED: True},
    )
    if not final_state[REPLACED]:
        msg = (
            "interval_classifier could not place the interval head: the base predictor has no Linear "
            "layer for the traverser to replace. Provide a ProbabilisticClassifier or "
            "LogitClassifier that ends in a linear layer."
        )
        raise ValueError(msg)
    return new_model


@predict.register(IntervalClassifierPredictor)
def _[**In](
    predictor: IntervalClassifierPredictor[In, ProbabilityIntervalsCredalSet],
    *args: In.args,
    **kwargs: In.kwargs,
) -> ProbabilityIntervalsCredalSet:
    """Wrap the predictor's packed interval output as a ``ProbabilityIntervalsCredalSet``."""
    return create_probability_intervals_from_lower_upper_array(predict_raw(predictor, *args, **kwargs))
