"""Shared credal net implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flextype import flexdispatch
from probly.method.method import predictor_transformation
from probly.predictor import LogitClassifier, Predictor, ProbabilisticClassifier, predict, predict_raw
from probly.representation.array_like import ArrayLike
from probly.traverse_nn import nn_compose
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse_with_state


@flexdispatch
def intersection_probability[T: ArrayLike](lower: T, upper: T) -> T:
    """Intersection probability of a probability interval based on :cite:`wang2024credalnet`.

    Maps an interval credal set ``[lower, upper]`` to a single probability
    vector via ``q_int_k = lower_k + alpha * (upper_k - lower_k)`` with
    ``alpha = (1 - sum(lower)) / sum(upper - lower)``. Assumes the interval
    is reachable, in which case ``alpha`` lies in ``[0, 1]`` and the result
    is a valid distribution. Used by the credal net's ``predict`` and the
    intersection-probability cross-entropy loss.
    """
    msg = f"No intersection_probability registered for array type {type(lower)}"
    raise NotImplementedError(msg)


@runtime_checkable
class CredalNetPredictor[**In, Out](Predictor[In, Out], Protocol):
    """Predictor that produces interval-valued probabilities, based on :cite:`wang2024credalnet`.

    The model's forward returns a packed ``(B, 2C)`` tensor with the lower
    and upper probability bounds for each class. ``predict_raw`` packs the
    user input before forwarding; ``predict`` returns the intersection
    probability (a single distribution per input); the credal-set view is
    available via the ``CredalNetRepresenter`` returned from
    :func:`probly.representer.representer`.
    """


REPLACED = GlobalVariable[bool]("REPLACED", default=False)

credal_net_traverser = flexdispatch_traverser[object](name="credal_net_traverser")


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier, LogitClassifier),
    preserve_predictor_type=False,
)  # ty:ignore[invalid-argument-type]
@CredalNetPredictor.register_factory
def credal_net[**In, Out](base: Predictor[In, Out]) -> CredalNetPredictor[In, Out]:
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
        ValueError: If ``base`` contains no ``nn.Linear`` (or the equivalent
            in another backend) for the credal head to replace.
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
def _[**In, Out](predictor: CredalNetPredictor[In, Out], *args: In.args, **kwargs: In.kwargs) -> Out:
    """Predict the intersection probability of the credal-net interval output.

    Returns a single probability vector per input (the paper's intersection
    probability of the predicted reachable interval). For the full credal-set
    view, use ``representer(predictor).predict(...)``.
    """
    raw = predict_raw(predictor, *args, **kwargs)
    n_classes = raw.shape[-1] // 2
    return intersection_probability(raw[..., :n_classes], raw[..., n_classes:])  # ty:ignore[invalid-return-type]
