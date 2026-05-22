"""Register torch-backend handlers for credal set plots."""

from __future__ import annotations

from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
    TorchDirichletLevelSetCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)

from ._binary import (
    _draw_credal_set_binary,
    _draw_distance_based_binary,
    _draw_intervals_binary,
    _draw_vertex_set_binary,
)
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

_draw_credal_set_binary.register(TorchProbabilityIntervalsCredalSet)(_draw_intervals_binary)
_draw_credal_set_binary.register(TorchDistanceBasedCredalSet)(_draw_distance_based_binary)
_draw_credal_set_binary.register(TorchDirichletLevelSetCredalSet)(_draw_distance_based_binary)
_draw_credal_set_binary.register(TorchConvexCredalSet)(_draw_vertex_set_binary)

_draw_credal_set_spider.register(TorchProbabilityIntervalsCredalSet)(_draw_intervals_spider)
_draw_credal_set_spider.register(TorchDistanceBasedCredalSet)(_draw_distance_based_spider)
_draw_credal_set_spider.register(TorchConvexCredalSet)(_draw_convex_set_spider)

_draw_credal_set_ternary.register(TorchProbabilityIntervalsCredalSet)(_draw_intervals)
_draw_credal_set_ternary.register(TorchDistanceBasedCredalSet)(_draw_distance_based)
_draw_credal_set_ternary.register(TorchConvexCredalSet)(_draw_convex_set)
