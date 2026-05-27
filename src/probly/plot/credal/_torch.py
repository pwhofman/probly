"""Register torch-backend handlers for credal set plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)

from ._binary import (
    _draw_credal_set_binary,
    _draw_distance_based_binary,
    _draw_intervals_binary,
    _draw_vertex_set_binary,
)
from ._data import _get_unnormalized_probabilities, _to_numpy
from ._spider import (
    _draw_convex_set_spider,
    _draw_credal_set_spider,
    _draw_distance_based_spider,
    _draw_intervals_spider,
)
from ._ternary import (
    _draw_convex_set,
    _draw_credal_set_ternary,
    _draw_distance_based,
    _draw_intervals,
)

if TYPE_CHECKING:
    import numpy as np

_draw_credal_set_binary.register(TorchProbabilityIntervalsCredalSet)(_draw_intervals_binary)
_draw_credal_set_binary.register(TorchDistanceBasedCredalSet)(_draw_distance_based_binary)
_draw_credal_set_binary.register(TorchConvexCredalSet)(_draw_vertex_set_binary)

_draw_credal_set_spider.register(TorchProbabilityIntervalsCredalSet)(_draw_intervals_spider)
_draw_credal_set_spider.register(TorchDistanceBasedCredalSet)(_draw_distance_based_spider)
_draw_credal_set_spider.register(TorchConvexCredalSet)(_draw_convex_set_spider)

_draw_credal_set_ternary.register(TorchProbabilityIntervalsCredalSet)(_draw_intervals)
_draw_credal_set_ternary.register(TorchDistanceBasedCredalSet)(_draw_distance_based)
_draw_credal_set_ternary.register(TorchConvexCredalSet)(_draw_convex_set)


@_get_unnormalized_probabilities.register(TorchConvexCredalSet)
def _convex_probabilities(data: TorchConvexCredalSet) -> np.ndarray:
    return _to_numpy(data.tensor.unnormalized_probabilities)


@_get_unnormalized_probabilities.register(TorchDistanceBasedCredalSet)
def _distance_based_probabilities(data: TorchDistanceBasedCredalSet) -> np.ndarray:
    return _to_numpy(data.nominal.unnormalized_probabilities)
