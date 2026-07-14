"""Shared masksembles implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, override, runtime_checkable

from flextype import flexdispatch

from probly.predictor import LogitClassifier, RandomPredictor, predict
from probly.representation.sample._common import Sample
from probly.representer._representer import Representer, representer
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import is_first_layer, nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from flextype.isinstance import LazyType

    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser


@runtime_checkable
class MasksemblesPredictor[**In, Out](RandomPredictor[In, Out], Protocol):
    """Protocol marking a predictor where binary masks were appended after hidden layers."""


@flexdispatch
def _wrap_masksembles_logits(sample: Sample) -> Sample:
    """Wrap a per-member logit sample as a typed distribution sample (default: passthrough)."""
    return sample


class MasksemblesRepresenter[**In, Out](Representer[Any, In, Out, Sample]):
    """Represent Masksembles predictions as a per-mask sample."""

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> Sample:
        """Forward through the predictor and return the per-mask sample."""
        return _wrap_masksembles_logits(predict(self.predictor, *args, **kwargs))


representer.register(MasksemblesPredictor, MasksemblesRepresenter)


N_MASKS = GlobalVariable[int]("N_MASKS", "Number of masks for Masksembles.")
SCALE = GlobalVariable[float]("SCALE", "Scale parameter controlling mask overlap.")

masksembles_traverser = flexdispatch_traverser[object](name="masksembles_traverser")


@flexdispatch
def _attach_n_masks(model: object, n_masks: int) -> None:
    """Attach the number of masks to the traversed model so it survives serialization."""
    msg = f"No n_masks attacher registered for type {type(model)}."
    raise NotImplementedError(msg)


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be appended by Masksembles layers."""
    masksembles_traverser.register(
        cls=cls,
        traverser=traverser,
        skip_if=is_first_layer,
        vars={
            "n_masks": N_MASKS,
            "scale": SCALE,
        },
    )


@predictor_transformation(permitted_predictor_types=[LogitClassifier], preserve_predictor_type=False)
@MasksemblesPredictor.register_factory
def masksembles[T: Predictor](
    base: T,
    n_masks: int = 4,
    scale: float = 2.0,
) -> T:
    """Create a Masksembles predictor from a base predictor based on :cite:`durasovMasksembles2021`.

    Appends a binary mask layer after each hidden linear or convolutional layer (the last
    layer is skipped). The result is tagged with ``n_masks`` so :func:`predict` can tile
    inputs and aggregate per-mask outputs as a :class:`~probly.representation.sample._common.Sample`.

    Args:
        base: The base model to apply Masksembles to.
        n_masks: Number of binary masks to generate.
        scale: Controls mask overlap; higher values produce less correlated masks at the
            cost of capacity per masked sub-network.

    Returns:
        The Masksembles predictor wrapping the base model.
    """
    transformed = traverse(
        base,
        nn_compose(masksembles_traverser),
        init={
            TRAVERSE_REVERSED: True,
            N_MASKS: n_masks,
            SCALE: scale,
            CLONE: True,
        },
    )
    _attach_n_masks(transformed, n_masks)
    return transformed
