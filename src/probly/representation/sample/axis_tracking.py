"""Implementation of axis tracking through NumPy-style indexing operations."""

from __future__ import annotations

from dataclasses import dataclass
from types import EllipsisType
from typing import TYPE_CHECKING

import numpy as np

from flextype import flexdispatch
from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR

if TYPE_CHECKING:
    from probly.representation.array_like import ToIndex, ToIndices


@dataclass(frozen=True, slots=True)
class ArrayIndex:
    """Marker for an advanced indexing array, which can be either integer or boolean and of any shape."""

    ndim: int
    is_boolean: bool


type _BasicIndexElement = slice | int | None
type _AdvancedIndexElement = int | bool | slice | ArrayIndex | EllipsisType
type _InternalIndexElement = _BasicIndexElement | _AdvancedIndexElement


@flexdispatch
def convert_idx(idx: ToIndex) -> _InternalIndexElement:  # noqa: PLR0911
    """Convert IndexElement to a _InternalIndexElement."""
    if isinstance(idx, slice) or idx is None or idx is Ellipsis:
        return idx
    if isinstance(idx, (bool, np.bool_)):
        return bool(idx)
    if isinstance(idx, (int, np.integer)):
        return int(idx)
    if hasattr(idx, "ndim") and hasattr(idx, "dtype"):
        # Use duck typing to support arbitrary array types without hard dependencies on specific libraries.
        # While we also support extension via singledispatch,
        # this fallback enables support for private array-likes, such as JAX tracers.
        if idx.ndim == 0:
            if idx.dtype == bool:
                return bool(idx)
            return 0  # use 0 as a sentinel for any 0d integer index
        return ArrayIndex(ndim=idx.ndim, is_boolean=idx.dtype == bool)  # ty:ignore[invalid-argument-type]

    idx = np.asanyarray(idx)
    if idx.ndim == 0:
        if idx.dtype == bool:
            return bool(idx)
        return 0  # use 0 as a sentinel for any 0d integer index
    return ArrayIndex(ndim=idx.ndim, is_boolean=idx.dtype == bool)


@convert_idx.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_axis_tracking  # noqa: F401, PLC0415


@convert_idx.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_axis_tracking  # noqa: F401, PLC0415


def _normalize_index(index: ToIndices, ndim: int, torch_indexing: bool = False) -> tuple[_InternalIndexElement, ...]:
    if not isinstance(index, tuple):
        normalized_index = (convert_idx(index),)
    else:
        normalized_index: tuple[_InternalIndexElement, ...] = tuple(convert_idx(idx) for idx in index)  # ty:ignore[invalid-argument-type]

    # Expand ellipsis
    for i, idx in enumerate(normalized_index):
        if idx is Ellipsis:
            before: tuple[_InternalIndexElement, ...] = normalized_index[:i]
            after: tuple[_InternalIndexElement, ...] = normalized_index[i + 1 :]
            consumed_before = 0
            consumed_after = 0
            for inner_idx in before:
                if isinstance(inner_idx, ArrayIndex) and inner_idx.is_boolean:
                    consumed_before += inner_idx.ndim
                elif inner_idx is not None and not isinstance(inner_idx, bool):
                    consumed_before += 1
            for inner_idx in after:
                if isinstance(inner_idx, ArrayIndex) and inner_idx.is_boolean:
                    consumed_after += inner_idx.ndim
                elif inner_idx is not None and not isinstance(inner_idx, bool):
                    consumed_after += 1
            missing = ndim - (consumed_before + consumed_after)
            if missing == 0 and not torch_indexing:
                # See https://github.com/pytorch/pytorch/pull/158297
                return (*before, idx, *after)
            return (*before, *(slice(None),) * missing, *after)

    # Pad with full slices if index is short
    if len(normalized_index) < ndim:
        return normalized_index + (slice(None),) * (ndim - len(normalized_index))

    return normalized_index


def _split_index(  # noqa: PLR0912
    index: tuple[_InternalIndexElement, ...],
    torch_indexing: bool = False,
) -> tuple[tuple[_BasicIndexElement, ...], tuple[_AdvancedIndexElement, ...]]:
    """Split index into basic and advanced index components.

    See https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    """
    int_index: dict[int, int] = {}
    basic_index: list[_BasicIndexElement] = []
    advanced_index: list[_AdvancedIndexElement] = []
    has_basic = False
    has_advanced = False

    for i, idx in enumerate(index):
        if isinstance(idx, int) and not isinstance(idx, bool):
            int_index[i] = idx
            basic_index.append(slice(None))
            advanced_index.append(slice(None))
        elif isinstance(idx, slice) or idx is None:
            basic_index.append(idx)
            advanced_index.append(slice(None))
            if idx != slice(None):
                has_basic = True
        elif idx is Ellipsis:
            advanced_index.append(idx)
            # Don't set has_advanced here since ellipsis is only relevant
            # for determining whether non-contiguous advanced indexing should
            # move the advanced indexing target axis to the fromt.
        else:
            basic_index.append(slice(None))
            advanced_index.append(idx)
            has_advanced = True

    if len(int_index) > 0:
        # Remove integer indexed axes from both basic and advanced indices
        if has_advanced and not torch_indexing:
            for i, idx in int_index.items():
                advanced_index[i] = idx
        else:
            for i, idx in int_index.items():
                basic_index[i] = idx
            if has_advanced:
                # In torch, integers are processed via basic indexing rules,
                # so the axes consumed by them have to be removed from the advanced index.
                advanced_index = [idx for i, idx in enumerate(advanced_index) if i not in int_index]
            has_basic = True

    if not has_advanced:
        return tuple(basic_index), ()

    if not has_basic:
        return (), tuple(advanced_index)

    return tuple(basic_index), tuple(advanced_index)


def _track_axis_basic(
    basic_index: tuple[_BasicIndexElement, ...],
    special_axis: int,
) -> int | None:
    """Track the special axis through basic indexing operations only.

    https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
    """
    new_axis = special_axis
    consumed_axes = 0

    for idx in basic_index:
        if isinstance(idx, int):
            if consumed_axes < special_axis:
                new_axis -= 1
            elif consumed_axes == special_axis:
                return None  # axis removed
            consumed_axes += 1
        elif idx is None:
            new_axis += 1
        else:
            consumed_axes += 1
        if consumed_axes > special_axis:
            break

    return new_axis


def _track_axis_advanced(  # noqa: C901, PLR0912, PLR0915
    advanced_index: tuple[_AdvancedIndexElement, ...],
    special_axis: int,
) -> int | None:
    """Track the special axis through both basic and advanced indexing operations.

    https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
    """
    new_special_axis: int = special_axis
    consumed_axes: int = 0
    indexing_target_axis: int = -2  # -2 represents "not set"
    after_advanced_indexing: bool = False
    num_new_axes: int = 0
    special_axis_indexed: bool = False

    for idx in advanced_index:
        if isinstance(idx, slice):
            if indexing_target_axis != -2:
                after_advanced_indexing = True
            consumed_axes += 1
        elif idx is Ellipsis:
            if indexing_target_axis != -2:
                after_advanced_indexing = True
        elif isinstance(idx, bool):
            num_new_axes = max(num_new_axes, 1)
            if indexing_target_axis == -2:
                indexing_target_axis = consumed_axes - 1
        elif isinstance(idx, int):
            if indexing_target_axis == -2:
                indexing_target_axis = consumed_axes
            elif after_advanced_indexing:
                indexing_target_axis = -1

            if consumed_axes < special_axis:
                new_special_axis -= 1
            elif consumed_axes == special_axis:
                return None  # axis removed
            consumed_axes += 1
        elif isinstance(idx, ArrayIndex):
            if indexing_target_axis == -2:
                indexing_target_axis = consumed_axes
            elif after_advanced_indexing:
                indexing_target_axis = -1

            if idx.is_boolean:
                num_new_axes = max(num_new_axes, 1)
                if consumed_axes + idx.ndim <= special_axis:
                    new_special_axis -= idx.ndim
                elif consumed_axes <= special_axis < consumed_axes + idx.ndim:
                    special_axis_indexed = True
                consumed_axes += idx.ndim
            else:
                num_new_axes = max(num_new_axes, idx.ndim)
                if consumed_axes < special_axis:
                    new_special_axis -= 1
                elif consumed_axes == special_axis:
                    special_axis_indexed = True
                consumed_axes += 1

    if special_axis_indexed:
        if num_new_axes > 1:
            return None  # multidimensional advanced indexing makes special axis ambiguous
        new_special_axis = 0 if indexing_target_axis == -1 else indexing_target_axis
    elif indexing_target_axis != -2 and indexing_target_axis < special_axis:
        new_special_axis += num_new_axes

    return new_special_axis


def track_axis(
    index: ToIndices,
    special_axis: int,
    ndim: int,
    torch_indexing: bool = False,
) -> int | None:
    """Track the new position of a 'special' axis after a NumPy-style __getitem__ indexing operation.

    Args:
        index: The indexing object used in arr[index].
            Can be a slice, int, None, list, ndarray, ellipsis, or a tuple of such.
        special_axis: Index of the axis to track (0-based) before indexing.
        ndim: Number of dimensions of the array before indexing.
        torch_indexing: Whether to apply PyTorch's mixed basic/advanced indexing rules instead of NumPy's.

    Returns:
        The new axis index of the special axis after indexing, or None if the
        axis is removed (e.g., via integer indexing).
    """
    # Handle structured arrays (field access)
    if isinstance(index, str) or (isinstance(index, list) and all(isinstance(i, str) for i in index)):
        return special_axis

    normalized_index = _normalize_index(index, ndim, torch_indexing=torch_indexing)
    basic_index, advanced_index = _split_index(normalized_index, torch_indexing=torch_indexing)

    new_axis: int | None = special_axis

    if len(basic_index) > 0:
        new_axis = _track_axis_basic(basic_index, special_axis)
        if new_axis is None:
            return None

    if len(advanced_index) > 0:
        new_axis = _track_axis_advanced(advanced_index, new_axis)

    return new_axis
