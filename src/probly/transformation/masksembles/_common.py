"""Shared masksembles implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from probly.predictor import RandomPredictor
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import is_first_layer, nn_compose
from pytraverse import CLONE, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from flextype.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser


@runtime_checkable
class MasksemblesPredictor[**In, Out](RandomPredictor[In, Out], Protocol):
    """A predictor that applies Masksembles uncertainty estimation."""


N_MASKS = GlobalVariable[int]("N_MASKS", "Number of masks for Masksembles.")
SCALE = GlobalVariable[float]("SCALE", "Scale parameter controlling mask overlap.")

masksembles_traverser = flexdispatch_traverser[object](name="masksembles_traverser")


def generate_masks_(m: int, n: int, scale: float) -> torch.Tensor:
    total_positions = int(m * scale)
    masks = torch.zeros(n, total_positions)
    for _ in range(n):
        idx = torch.randperm(total_positions)[:m]
        masks[_, idx] = 1
    masks = masks[:, ~(masks == 0).all(dim=0)]
    return masks


def generate_masks(m: int, n: int, scale: float) -> torch.Tensor:
    masks = generate_masks_(m, n, scale)
    expected_size = int(m * scale * (1 - (1 - 1 / scale) ** n))
    while masks.shape[1] != expected_size:
        masks = generate_masks_(m, n, scale)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> torch.Tensor:
    m = int(int(c) / (scale * (1 - (1 - 1 / scale) ** n)))
    max_iter = 1000

    lo = max([scale * 0.8, 1.0])
    hi = scale * 1.2

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        masks = generate_masks(m, n, mid)
        if masks.shape[-1] == c:
            break
        if masks.shape[-1] > c:
            hi = mid
        else:
            lo = mid

    if masks.shape[-1] != c:
        msg = "generation_wrapper was unable to generate fitting masks"
        raise ValueError(msg)
    return masks


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


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=True)
@MasksemblesPredictor.register_factory
def masksembles[T: Predictor](
    base: T,
    n_masks: int = 4,
    scale: float = 2.0,
) -> T:
    """Create a Masksembles predictor from a base predictor.

    Based on [Masksembles for Uncertainty Estimation](https://arxiv.org/abs/2012.08334)

    Args:
        base: The base model to apply Masksembles to.
        n_masks: Number of binary masks to generate. Higher = more ensemble-like.
        scale: Scale parameter controlling mask overlap. Higher = less correlation.
        rng_collection: Optional rng collection name for flax layer initialization.
        rngs: Optional rngs for flax layer initialization.

    Returns:
        The Masksembles predictor wrapping the base model.
    """
    if n_masks < 1:
        msg = f"n_masks must be >= 1, got {n_masks}"
        raise ValueError(msg)
    if scale <= 0:
        msg = f"scale must be > 0, got {scale}"
        raise ValueError(msg)

    return traverse(
        base,
        nn_compose(masksembles_traverser),
        init={
            N_MASKS: n_masks,
            SCALE: scale,
            CLONE: True,
        },
    )
