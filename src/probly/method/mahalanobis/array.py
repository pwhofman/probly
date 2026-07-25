"""Numpy implementation of the Mahalanobis OOD representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution

from ._common import (
    MahalanobisRepresentation,
    combine_layer_scores,
    create_mahalanobis_representation,
)


@create_mahalanobis_representation.register(ArrayCategoricalDistribution)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayMahalanobisRepresentation(MahalanobisRepresentation, ArrayAxisProtected[np.ndarray]):
    """Mahalanobis representation backed by numpy arrays.

    ``weight`` and ``bias`` are shared (non per-sample) combiner parameters and
    are therefore left out of ``protected_axes`` so they ride along unchanged
    through indexing and batching.
    """

    softmax: ArrayCategoricalDistribution
    layer_scores: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"softmax": 0, "layer_scores": 1}


@combine_layer_scores.register(np.ndarray)
def array_combine_layer_scores(layer_scores: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Combine per-layer Mahalanobis confidences into a single OOD score.

    The combination is the logistic-regression logit ``s @ w + b``. With the
    default weights (``-1`` per layer and zero bias) this is the negated sum of
    the per-layer confidences, so a far-from-centroid (out-of-distribution) input
    yields a high score.
    """
    return layer_scores @ weight + bias
