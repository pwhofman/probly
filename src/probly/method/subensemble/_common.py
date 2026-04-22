"""Shared subensemble implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.method.ensemble import EnsemblePredictor
from probly.method.method import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


@runtime_checkable
class SubensemblePredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """Protocol for subensemble predictors."""

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


@flexdispatch
def subensemble_generator[**In, H, Out](
    base: Predictor[In, H],
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> SubensemblePredictor[In, Out]:
    """Generate a subensemble from a base model."""
    msg = f"No subensemble generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@SubensemblePredictor.register_factory
def subensemble[**In, H, Out](
    base: Predictor[In, H],
    num_heads: int,
    head: Predictor[[H], Out] | None = None,
    reset_params: bool = True,
    head_layer: int = 1,
) -> SubensemblePredictor[In, Out]:
    """Create a subensemble predictor from a base model or a base model and head model.

    Supported configurations:
        1. base only:
            Head is created by extracting the last `head_layer` layers of `base`,
            while the remaining layers are used to create the backbone.

        2. base and head:
            `base` is used as the shared backbone,
            `head` is duplicated `num_heads` times to form the subensemble heads.

    Args:
        base: Predictor, The model to be used as backbone or to create the backbone and heads.
        num_heads: int, The number of heads in the subensemble.
        head: Predictor, Optional model to be used as head of the subensemble.
        reset_params: bool, Whether to reset the parameters of each head.
        head_layer: int, Optional the number of layers used to create the head if no head model is provided.

    Returns:
        SubensemblePredictor, The subensemble predictor.

    Raises:
        ValueError: If `head_layer` or `num_heads` is not a positive integer.
    """
    if head_layer <= 0:
        msg = f"head_layer must be a positive number, but got head_layer={head_layer} instead."
        raise ValueError(msg)
    if num_heads <= 0:
        msg = f"num_heads must be positive number, but got num_heads={num_heads} instead."
        raise ValueError(msg)
    return subensemble_generator(base, num_heads=num_heads, head=head, reset_params=reset_params, head_layer=head_layer)
