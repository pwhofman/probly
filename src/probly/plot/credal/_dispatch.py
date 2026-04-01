"""Type-based dispatch for extracting metadata from credal sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_dispatch import lazydispatch
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)

if TYPE_CHECKING:
    type ArrayCredalSet = (
        ArrayProbabilityIntervalsCredalSet
        | ArrayDistanceBasedCredalSet
        | ArrayConvexCredalSet
        | ArrayDiscreteCredalSet
        | ArraySingletonCredalSet
    )


@lazydispatch
def _get_num_classes(data: ArrayCredalSet) -> int:
    """Extract the number of classes from any Array credal set type.

    Args:
        data: Any Array credal set instance.

    Returns:
        The number of classes.

    Raises:
        NotImplementedError: If the credal set type is not supported.
    """
    msg = f"Unsupported credal set type: {type(data).__name__}"
    raise NotImplementedError(msg)


@_get_num_classes.register(ArrayProbabilityIntervalsCredalSet)
def _(data: ArrayProbabilityIntervalsCredalSet) -> int:
    return data.num_classes


@_get_num_classes.register(ArrayDistanceBasedCredalSet)
def _(data: ArrayDistanceBasedCredalSet) -> int:
    return data.nominal.shape[-1]


@_get_num_classes.register(ArrayConvexCredalSet)
@_get_num_classes.register(ArrayDiscreteCredalSet)
@_get_num_classes.register(ArraySingletonCredalSet)
def _(data: ArrayConvexCredalSet | ArrayDiscreteCredalSet | ArraySingletonCredalSet) -> int:
    return data.array.shape[-1]
