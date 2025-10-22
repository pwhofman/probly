"""Tests for the generic traverser module.

This module tests the generic traverser functionality for standard Python
data types including tuples, lists, dictionaries, and sets.
"""

from __future__ import annotations

from typing import Any

from pytraverse.core import GlobalVariable, State, TraverserResult
from pytraverse.generic import (
    CLONE,
    TRAVERSE_KEYS,
    TRAVERSE_REVERSED,
    _dict_traverser,
    _list_traverser,
    _set_traverser,
    _tuple_traverser,
    generic_traverser,
)


def double_traverse(
    obj: Any,  # noqa: ANN401
    state: State,
    meta: Any = None,  # noqa: ANN401
    traverser: Any = None,  # noqa: ANN401
) -> TraverserResult:
    return obj * 2, state


def identity_traverse(
    obj: Any,  # noqa: ANN401
    state: State,
    meta: Any = None,  # noqa: ANN401
    traverser: Any = None,  # noqa: ANN401
) -> TraverserResult:
    return obj, state


class TestGenericTraverser:
    """Test the main generic traverser dispatch functionality."""

    def test_tuple_dispatch(self) -> None:
        """Test that tuples are dispatched to the tuple traverser."""
        test_tuple = (1, 2, 3)
        state: State = State()

        result, _ = generic_traverser(test_tuple, state, double_traverse)
        assert result == (2, 4, 6)
        assert isinstance(result, tuple)

    def test_list_dispatch(self) -> None:
        """Test that lists are dispatched to the list traverser."""
        test_list = [1, 2, 3]
        state: State = State()

        result, _ = generic_traverser(test_list, state, double_traverse)
        assert result == [2, 4, 6]
        assert isinstance(result, list)

    def test_dict_dispatch(self) -> None:
        """Test that dicts are dispatched to the dict traverser."""
        test_dict = {"a": 1, "b": 2}
        state: State = State()

        result, _ = generic_traverser(test_dict, state, double_traverse)
        assert result == {"a": 2, "b": 4}
        assert isinstance(result, dict)

    def test_set_dispatch(self) -> None:
        """Test that sets are dispatched to the set traverser."""
        test_set = {1, 2, 3}
        state: State = State()

        result, _ = generic_traverser(test_set, state, double_traverse)
        assert result == {2, 4, 6}
        assert isinstance(result, set)

    def test_custom_class_with_generic_fallback(self) -> None:
        """Test that unknown types fall back to identity traversal."""

        class CustomClass:
            def __init__(self, value: Any) -> None:  # noqa: ANN401
                self.value = value

        custom_obj = CustomClass(42)
        state: State = State()

        result, _ = generic_traverser(custom_obj, state, identity_traverse)
        assert result == custom_obj
        assert result is custom_obj  # Should be identity


class TestTupleTraverser:
    """Test tuple traversal functionality."""

    def test_basic_tuple_traversal(self) -> None:
        """Test basic tuple element traversal."""
        test_tuple = (1, "hello", 3.14)
        state: State = State()

        result, _ = _tuple_traverser(test_tuple, state, identity_traverse)
        assert result == (1, "hello", 3.14)
        assert isinstance(result, tuple)
        assert result is not test_tuple  # Always creates new tuple

    def test_empty_tuple(self) -> None:
        """Test traversal of empty tuple."""
        test_tuple = ()
        state: State = State()

        result, _ = _tuple_traverser(test_tuple, state, identity_traverse)
        assert result == ()
        assert isinstance(result, tuple)

    def test_nested_tuple_traversal(self) -> None:
        """Test traversal with meta information (indices)."""
        test_tuple = ("a", "b", "c")
        state: State = State()
        captured_metas = []

        def capture_meta_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            captured_metas.append(meta)
            return obj.upper(), state

        result, _ = _tuple_traverser(test_tuple, state, capture_meta_traverse)
        assert result == ("A", "B", "C")
        assert captured_metas == [0, 1, 2]

    def test_reverse_tuple_traversal(self) -> None:
        """Test tuple traversal with reversed order."""
        test_tuple = (1, 2, 3)
        state: State = State()
        state = TRAVERSE_REVERSED.set(state, True)
        captured_objs = []

        def capture_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            captured_objs.append(obj)
            return obj + 5, state

        result, _ = _tuple_traverser(test_tuple, state, capture_traverse)
        assert result == (6, 7, 8)
        assert captured_objs == [3, 2, 1]  # Should capture in reverse order


class TestListTraverser:
    """Test list traversal functionality."""

    def test_list_clone_enabled(self) -> None:
        """Test list traversal with cloning enabled (default)."""
        test_list = [1, 2, 3]
        state: State = State()
        state = CLONE.set(state, True)

        result, _ = _list_traverser(test_list, state, double_traverse)
        assert result == [2, 4, 6]
        assert result is not test_list  # New list created
        assert test_list == [1, 2, 3]  # Original unchanged

    def test_list_clone_disabled(self) -> None:
        """Test list traversal with cloning disabled."""
        test_list = [1, 2, 3]
        state: State = State()
        state = CLONE.set(state, False)

        result, _ = _list_traverser(test_list, state, double_traverse)
        assert result == [2, 4, 6]
        assert result is test_list  # Same list modified
        assert test_list == [2, 4, 6]  # Original modified

    def test_empty_list(self) -> None:
        """Test traversal of empty list."""
        test_list: list = []
        state: State = State()

        result, _ = _list_traverser(test_list, state, identity_traverse)
        assert result == []
        assert isinstance(result, list)

    def test_list_with_meta(self) -> None:
        """Test list traversal captures indices as meta."""
        test_list = [0, 1, 2]
        state: State = State()
        captured_metas = []

        def capture_meta_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            assert obj == meta
            captured_metas.append(meta)
            return obj, state

        result, _ = _list_traverser(test_list, state, capture_meta_traverse)
        assert result == test_list
        assert captured_metas == [0, 1, 2]

    def test_list_subclass_preserved(self) -> None:
        """Test that list subclasses are preserved."""

        class CustomList(list):
            def __init__(self, *args: Any) -> None:  # noqa: ANN401
                super().__init__(*args)

        test_list = CustomList([1, 2, 3])
        state: State = State()
        state = CLONE.set(state, True)

        result, _ = _list_traverser(test_list, state, identity_traverse)
        assert test_list is not result
        assert isinstance(result, CustomList)
        assert result == [1, 2, 3]

    def test_reverse_list_traversal(self) -> None:
        """Test list traversal with reversed order."""
        test_list = [1, 2, 3]
        state: State = State()
        state = TRAVERSE_REVERSED.set(state, True)
        captured_objs = []

        def capture_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            captured_objs.append(obj)
            return obj + 5, state

        result, _ = _list_traverser(test_list, state, capture_traverse)
        assert result == [6, 7, 8]
        assert captured_objs == [3, 2, 1]

    def test_reverse_list_traversal_clone_disabled(self) -> None:
        """Test reversed list traversal with cloning disabled."""
        test_list = [1, 2, 3]
        state: State = State()
        state = TRAVERSE_REVERSED.set(state, True)
        state = CLONE.set(state, False)
        captured_objs = []

        def capture_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            captured_objs.append(obj)
            return obj + 5, state

        result, _ = _list_traverser(test_list, state, capture_traverse)
        assert result == [6, 7, 8]
        assert captured_objs == [3, 2, 1]
        assert result is test_list


class TestDictTraverser:
    """Test dictionary traversal functionality."""

    def test_dict_values_only(self) -> None:
        """Test dictionary traversal of values only (default)."""
        test_dict = {"a": 1, "b": 2, "c": 3}
        state: State = State()
        state = TRAVERSE_KEYS.set(state, False)
        state = CLONE.set(state, True)

        result, _ = _dict_traverser(test_dict, state, double_traverse)
        assert result == {"a": 2, "b": 4, "c": 6}
        assert result is not test_dict

    def test_dict_traverse_keys(self) -> None:
        """Test dictionary traversal including keys."""
        test_dict = {"first": 1, "second": 2}
        state: State = State()
        state = TRAVERSE_KEYS.set(state, True)

        def upper_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            if isinstance(obj, str):
                return obj.upper(), state
            return obj * 2, state

        result, _ = _dict_traverser(test_dict, state, upper_traverse)
        assert result == {"FIRST": 2, "SECOND": 4}

    def test_dict_clone_disabled(self) -> None:
        """Test in-place dictionary modification."""
        test_dict = {"a": 1, "b": 2}
        state: State = State()
        state = TRAVERSE_KEYS.set(state, False)
        state = CLONE.set(state, False)

        result, _ = _dict_traverser(test_dict, state, double_traverse)
        assert result == {"a": 2, "b": 4}
        assert result is test_dict  # Same dict modified
        assert test_dict == {"a": 2, "b": 4}

    def test_dict_traverse_keys_forces_clone(self) -> None:
        """Test that traversing keys forces cloning even when CLONE=False."""
        test_dict = {"a": 1, "b": 2}
        state: State = State()
        state = TRAVERSE_KEYS.set(state, True)
        state = CLONE.set(state, False)

        result, _ = _dict_traverser(test_dict, state, identity_traverse)
        assert result == {"a": 1, "b": 2}
        assert result is not test_dict  # Forced to clone
        assert isinstance(result, dict)

    def test_empty_dict(self) -> None:
        """Test traversal of empty dictionary."""
        test_dict: dict = {}
        state: State = State()

        result, _ = _dict_traverser(test_dict, state, identity_traverse)
        assert result == {}
        assert isinstance(result, dict)

    def test_dict_with_meta(self) -> None:
        """Test that dict values receive keys as meta."""
        test_dict = {"first": 1, "second": 2}
        state: State = State()
        state = TRAVERSE_KEYS.set(state, False)

        def capture_meta_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            return f"{meta}={obj}", state

        result, _ = _dict_traverser(test_dict, state, capture_meta_traverse)
        assert result == {"first": "first=1", "second": "second=2"}

    def test_dict_subclass_preserved(self) -> None:
        """Test that dict subclasses are preserved."""

        class CustomDict(dict):
            def __init__(self, *args: Any) -> None:  # noqa: ANN401
                super().__init__(*args)

        test_dict = CustomDict({"a": 1, "b": 2})
        state: State = State()
        state = CLONE.set(state, True)

        result, _ = _dict_traverser(test_dict, state, identity_traverse)
        assert isinstance(result, CustomDict)
        assert result is not test_dict  # New dict created
        assert result == {"a": 1, "b": 2}

    def test_reverse_dict_traversal(self) -> None:
        """Test dictionary traversal with reversed order."""
        test_dict = {"a": 1, "b": 2, "c": 3}
        state: State = State()
        state = TRAVERSE_REVERSED.set(state, True)
        captured_objs = []

        def capture_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            captured_objs.append(obj)
            return obj + 5, state

        result, _ = _dict_traverser(test_dict, state, capture_traverse)
        assert isinstance(result, dict)
        assert result == {"a": 6, "b": 7, "c": 8}
        assert list(result.keys()) == ["a", "b", "c"]  # Order preserved
        assert captured_objs == [3, 2, 1]

    def test_reverse_dict_traversal_clone_disabled(self) -> None:
        """Test reversed dictionary traversal with cloning disabled."""
        test_dict = {"a": 1, "b": 2, "c": 3}
        state: State = State()
        state = TRAVERSE_REVERSED.set(state, True)
        state = CLONE.set(state, False)
        captured_objs = []

        def capture_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            captured_objs.append(obj)
            return obj + 5, state

        result, _ = _dict_traverser(test_dict, state, capture_traverse)
        assert isinstance(result, dict)
        assert result == {"a": 6, "b": 7, "c": 8}
        assert list(result.keys()) == ["a", "b", "c"]
        assert result is test_dict  # Same dict modified


class TestSetTraverser:
    """Test set traversal functionality."""

    def test_basic_set_traversal(self) -> None:
        """Test basic set element traversal."""
        test_set = {1, 2, 3}
        state: State = State()

        result, _ = _set_traverser(test_set, state, double_traverse)
        assert result == {2, 4, 6}
        assert isinstance(result, set)

    def test_empty_set(self) -> None:
        """Test traversal of empty set."""
        test_set: set = set()
        state: State = State()

        result, _ = _set_traverser(test_set, state, identity_traverse)
        assert result == set()
        assert isinstance(result, set)

    def test_set_with_duplicates_after_traversal(self) -> None:
        """Test set behavior when traversal creates duplicates."""
        test_set = {1, 2, 3}
        state: State = State()

        def modulo_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            return obj % 2, state  # Maps to 0 or 1

        result, _ = _set_traverser(test_set, state, modulo_traverse)
        assert result == {0, 1}  # Duplicates removed
        assert isinstance(result, set)

    def test_set_subclass_preserved(self) -> None:
        """Test that set subclasses are preserved."""

        class CustomSet(set):
            def __init__(self, *args: Any) -> None:  # noqa: ANN401
                super().__init__(*args)

        test_set = CustomSet({1, 2, 3})
        state: State = State()

        result, _ = _set_traverser(test_set, state, identity_traverse)
        assert isinstance(result, CustomSet)
        assert result is not test_set  # New set created
        assert result == {1, 2, 3}


class TestIntegrationScenarios:
    """Test complex integration scenarios."""

    def test_nested_data_structures(self) -> None:
        """Test traversal of deeply nested data structures."""
        nested_data = {
            "list": [2, 3, {"inner": 4}],
            "tuple": (5, 6, {7, 8}),
            "set": {9, 10},
        }

        state: State = State()

        def increment_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            if isinstance(obj, int):
                return obj + 1, state
            # For non-int objects, delegate to the generic traverser
            return generic_traverser(obj, state, increment_traverse)

        result, _ = generic_traverser(nested_data, state, increment_traverse)

        expected = {
            "list": [3, 4, {"inner": 5}],
            "tuple": (6, 7, {8, 9}),
            "set": {10, 11},
        }
        assert result == expected

    def test_state_threading(self) -> None:
        """Test that state is properly threaded through traversal."""
        counter_var = GlobalVariable[int]("counter", default=0)
        test_list = [1, 2, 3]
        state: State = State()
        state = counter_var.set(state, 0)

        def counting_traverse(
            obj: Any,  # noqa: ANN401
            state: State,
            meta: Any = None,  # noqa: ANN401
            traverser: Any = None,  # noqa: ANN401
        ) -> TraverserResult:
            current_count = state[counter_var]
            state[counter_var] += 1
            return obj * current_count, state

        result, final_state = _list_traverser(test_list, state, counting_traverse)
        assert result == [0, 2, 6]  # [1*0, 2*1, 3*2]
        assert counter_var.get(final_state) == 3
