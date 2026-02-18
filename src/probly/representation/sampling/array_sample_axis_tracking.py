"""Implementation of axis tracking through NumPy-style indexing operations."""

from __future__ import annotations

from collections.abc import Sequence
from types import EllipsisType
from typing import Any

import numpy as np
import numpy.typing as npt

type IntSeq = Sequence[int | IntSeq]
type BoolSeq = Sequence[bool | BoolSeq]

type BasicIndexElement = slice | int | np.integer | None
type AdvancedIndexElement = int | np.integer | slice | npt.NDArray[Any]
type IndexElement = BasicIndexElement | AdvancedIndexElement | EllipsisType | IntSeq | BoolSeq

type Index = IndexElement | tuple[IndexElement, ...]


def _normalize_index(index: Index, ndim: int) -> tuple[IndexElement, ...]:
    if not isinstance(index, tuple):
        index = (index,)

    # Expand ellipsis
    for i, idx in enumerate(index):
        if idx is Ellipsis:
            before = index[:i]
            after = index[i + 1 :]
            missing = ndim - (i + len(after))
            index = before + (slice(None),) * missing + after
            break

    # Pad with full slices if index is short
    if len(index) < ndim:
        index = index + (slice(None),) * (ndim - len(index))

    return index


def _convert_idx(idx: IndexElement) -> np.ndarray | int:
    """Convert index elements to a numpy array or an integer."""
    if isinstance(idx, np.ndarray):
        if idx.ndim == 0:
            return idx.__index__()
        return idx
    try:
        return idx.__index__()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        idx = np.asanyarray(idx)
        if idx.ndim == 0:
            return idx.__index__()
        return idx


def _split_index(  # noqa: C901, PLR0912
    index: tuple[IndexElement, ...],
) -> tuple[tuple[BasicIndexElement, ...], tuple[AdvancedIndexElement, ...]]:
    """Split index into basic and advanced index components.

    See https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    """
    int_index: dict[int, np.integer | int] = {}
    basic_index: list[BasicIndexElement] = []
    advanced_index: list[AdvancedIndexElement] = []
    has_basic = False
    has_advanced = False

    for i, idx in enumerate(index):
        if isinstance(idx, (np.integer, int)):
            int_index[i] = idx
            basic_index.append(slice(None))
            advanced_index.append(slice(None))
        elif isinstance(idx, slice) or idx is None:
            basic_index.append(idx)
            advanced_index.append(slice(None))
            has_basic = True
        else:
            converted_idx = _convert_idx(idx)
            if isinstance(converted_idx, int):
                int_index[i] = converted_idx
                basic_index.append(slice(None))
                advanced_index.append(slice(None))
            else:
                basic_index.append(slice(None))
                advanced_index.append(converted_idx)
                has_advanced = True

    if len(int_index) > 0:
        # Remove integer indexed axes from both basic and advanced indices
        if has_advanced:
            for i, idx in int_index.items():
                advanced_index[i] = idx
        else:
            for i, idx in int_index.items():
                basic_index[i] = idx

    if not has_advanced:
        return tuple(basic_index), ()

    if not has_basic:
        return (), tuple(advanced_index)

    return tuple(basic_index), tuple(advanced_index)


def _track_axis_basic(
    basic_index: tuple[BasicIndexElement, ...],
    special_axis: int,
) -> int | None:
    """Track the special axis through basic indexing operations only.

    https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
    """
    new_axis = special_axis
    consumed_axes = 0

    for idx in basic_index:
        if isinstance(idx, (np.integer, int)):
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


def _track_axis_advanced(  # noqa: C901, PLR0912
    advanced_index: tuple[AdvancedIndexElement, ...],
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
        elif isinstance(idx, (np.integer, int)):
            if indexing_target_axis == -2:
                indexing_target_axis = consumed_axes
            elif after_advanced_indexing:
                indexing_target_axis = -1

            if consumed_axes < special_axis:
                new_special_axis -= 1
            elif consumed_axes == special_axis:
                return None  # axis removed
            consumed_axes += 1
        elif isinstance(idx, np.ndarray):
            if indexing_target_axis == -2:
                indexing_target_axis = consumed_axes
            elif after_advanced_indexing:
                indexing_target_axis = -1

            if idx.dtype == bool:
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
        new_special_axis = indexing_target_axis
    elif indexing_target_axis != -2 and indexing_target_axis < special_axis:
        new_special_axis += num_new_axes

    return new_special_axis


def track_axis(
    index: Index,
    special_axis: int,
    ndim: int,
) -> int | None:
    """Track the new position of a 'special' axis after a NumPy-style __getitem__ indexing operation.

    Args:
        index: The indexing object used in arr[index].
            Can be a slice, int, None, list, ndarray, ellipsis, or a tuple of such.
        special_axis: Index of the axis to track (0-based) before indexing.
        ndim: Number of dimensions of the array before indexing.

    Returns:
        The new axis index of the special axis after indexing, or None if the
        axis is removed (e.g., via integer indexing).
    """
    # Handle structured arrays (field access)
    if isinstance(index, str) or (isinstance(index, list) and all(isinstance(i, str) for i in index)):
        return special_axis

    index = _normalize_index(index, ndim)
    basic_index, advanced_index = _split_index(index)

    new_axis: int | None = special_axis

    if len(basic_index) > 0:
        new_axis = _track_axis_basic(basic_index, special_axis)
        if new_axis is None:
            return None

    if len(advanced_index) > 0:
        new_axis = _track_axis_advanced(advanced_index, new_axis)  # type: ignore[arg-type]

    return new_axis
