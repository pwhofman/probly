"""Tests for probly.traverse.composition module."""

from __future__ import annotations

from typing import Any, Union

import pytest

from pytraverse import Traverser
from pytraverse.composition import (
    SingledispatchTraverser,
    _is_union_type,
    _is_valid_dispatch_type,
    sequential,
    top_sequential,
)
from pytraverse.core import State, TraverserCallback, TraverserResult


def dummy_traverser(
    obj: Any,  # noqa: ANN401
    state: State[Any],
    traverse: TraverserCallback,
) -> TraverserResult:
    return obj, state  # pragma: no cover


def dummy_traverse(
    obj: Any,  # noqa: ANN401
    state: State[Any],
    meta: Any = None,  # noqa: ANN401
    traverser: Any = None,  # noqa: ANN401
) -> TraverserResult:
    return obj, state


def test_sequential_basic() -> None:
    """Test basic sequential composition of traversers."""

    def add_one(
        obj: int,
        state: State[int],
        traverse: TraverserCallback[int],
    ) -> TraverserResult[int]:
        return obj + 1, state

    def multiply_two(
        obj: int,
        state: State[int],
        traverse: TraverserCallback[int],
    ) -> TraverserResult[int]:
        return obj * 2, state

    composed: Traverser[int] = sequential(add_one, multiply_two)
    state: State[int] = State()

    result, new_state = composed(5, state, dummy_traverse)
    assert result == 12  # (5 + 1) * 2 = 12
    assert new_state is state


def test_sequential_with_name() -> None:
    """Test sequential composition with custom name."""
    composed: Traverser[Any] = sequential(dummy_traverser, name="test_traverser")
    assert composed.__name__ == "test_traverser"
    assert "test_traverser" in composed.__qualname__


def test_sequential_empty() -> None:
    """Test sequential with no traversers."""
    composed: Traverser = sequential()
    state: State[int] = State()

    result, new_state = composed(42, state, dummy_traverse)
    assert result == 42
    assert new_state is state


def test_top_sequential_basic() -> None:
    """Test basic top_sequential composition."""

    def preprocessor(
        obj: str,
        state: State[str],
        traverse: TraverserCallback[str],
    ) -> TraverserResult[str]:
        processed, state = traverse(obj.upper(), state)
        return processed, state

    def processor(
        obj: str,
        state: State[str],
        traverse: TraverserCallback[str],
    ) -> TraverserResult[str]:
        if not obj.startswith("processed_"):
            obj = f"processed_{obj}"
        return traverse(obj, state)

    composed = top_sequential(preprocessor, processor)
    state: State[str] = State()

    result, new_state = composed("hello", state, dummy_traverse)
    assert result == "processed_HELLO"
    assert new_state is state


def test_top_sequential_with_name() -> None:
    """Test top_sequential with custom name."""
    composed = top_sequential(dummy_traverser, name="top_test")
    assert composed.__name__ == "top_test"
    assert "top_test" in composed.__qualname__


def test_is_union_type() -> None:
    """Test _is_union_type helper function."""
    # Test Union types
    assert _is_union_type(Union[int, str]) is True  # noqa: UP007
    assert _is_union_type(int | str) is True  # Python 3.10+ union syntax

    # Test non-union types
    assert _is_union_type(int) is False
    assert _is_union_type(str) is False
    assert _is_union_type(list) is False


def test_is_valid_dispatch_type() -> None:
    """Test _is_valid_dispatch_type helper function."""
    # Test regular types
    assert _is_valid_dispatch_type(int) is True
    assert _is_valid_dispatch_type(str) is True
    assert _is_valid_dispatch_type(list) is True

    # Test union of types
    assert _is_valid_dispatch_type(Union[int, str]) is True  # noqa: UP007
    assert _is_valid_dispatch_type(int | str) is True

    # Test invalid types (not actual types)
    assert _is_valid_dispatch_type("not_a_type") is False


class TestSingledispatchTraverser:
    """Test cases for SingledispatchTraverser class."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        traverser = SingledispatchTraverser[int]()
        assert hasattr(traverser, "_dispatch")
        assert callable(traverser)

    def test_init_with_name(self) -> None:
        """Test initialization with custom name."""
        traverser = SingledispatchTraverser[int](name="test_dispatcher")
        assert traverser.__name__ == "test_dispatcher"
        assert "test_dispatcher" in traverser.__qualname__

    def test_init_with_default_traverser(self) -> None:
        """Test initialization with a default traverser function."""

        def default_func(obj: int, traverse: Any, **kwargs: Any) -> int:  # noqa: ANN401
            return obj * 2

        traverser = SingledispatchTraverser[int](default_func)
        # Should inherit name from the function
        assert traverser.__name__ == "default_func"

    def test_call_default(self) -> None:
        """Test calling the traverser with default behavior."""
        traverser = SingledispatchTraverser[int]()
        state: State[int] = State()

        result, new_state = traverser(42, state, dummy_traverse)
        assert result == 42  # Identity traverser behavior
        assert new_state is state

    def test_register_for_type(self) -> None:
        """Test registering a traverser for a specific type."""
        traverser = SingledispatchTraverser[Any]()

        @traverser.register(int)
        def int_handler(
            obj: int,
            state: State[Any],
            traverse: TraverserCallback[Any],
        ) -> TraverserResult[Any]:
            return obj * 2, state

        state: State[Any] = State()

        # Test with int
        result, new_state = traverser(5, state, dummy_traverse)
        assert result == 10

        # Test with string (should use default)
        result, new_state = traverser("hello", state, dummy_traverse)
        assert result == "hello"

    def test_register_multiple_types(self) -> None:
        """Test registering traversers for multiple types."""
        traverser = SingledispatchTraverser[Any]()

        @traverser.register(list)
        def list_handler(
            obj: list,
            state: State[Any],
            traverse: TraverserCallback[Any],
        ) -> TraverserResult[Any]:
            return [x * 2 for x in obj], state

        @traverser.register(dict)
        def dict_handler(
            obj: dict,
            state: State[Any],
            traverse: TraverserCallback[Any],
        ) -> TraverserResult[Any]:
            return {k: v * 2 for k, v in obj.items()}, state

        state: State[Any] = State()

        # Test list
        result, new_state = traverser([1, 2, 3], state, dummy_traverse)
        assert result == [2, 4, 6]

        # Test dict
        result, new_state = traverser({"a": 1, "b": 2}, state, dummy_traverse)
        assert result == {"a": 2, "b": 4}

    def test_register_direct_call(self) -> None:
        """Test registering via direct method call."""
        traverser = SingledispatchTraverser[Any]()

        def str_handler(
            obj: str,
            state: State[Any],
            traverse: TraverserCallback[Any],
        ) -> TraverserResult[Any]:
            return obj.upper(), state

        # Register directly
        traverser.register(str, str_handler)

        state: State[Any] = State()

        result1, _ = traverser(1, state, dummy_traverse)
        result2, _ = traverser("hello", state, dummy_traverse)
        assert result1 == 1
        assert result2 == "HELLO"

    def test_register_as_default(self) -> None:
        """Test registering a function as the default."""

        def custom_default(
            obj: object,
            state: State[Any],
            traverse: TraverserCallback[Any],
        ) -> TraverserResult[Any]:
            return f"default_{obj}", state

        traverser = SingledispatchTraverser[Any](custom_default)

        @traverser.register
        def custom_str(
            obj: str,
            state: State[Any],
            traverse: TraverserCallback[Any],
        ) -> TraverserResult[Any]:
            return f"str_{obj}", state

        state: State[Any] = State()

        # Should use custom default for unregistered types
        result, _ = traverser(42, state, dummy_traverse)
        assert result == "default_42"
        result, _ = traverser("a", state, dummy_traverse)
        assert result == "str_a"

    def test_register_invalid_type_error(self) -> None:
        """Test that registering with invalid type raises TypeError."""
        traverser = SingledispatchTraverser[Any]()

        def dummy_func(
            obj: Any,  # noqa: ANN401
            traverse: Any,  # noqa: ANN401
            **kwargs: Any,  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            return obj  # pragma: no cover

        # This should raise TypeError for invalid first argument
        with pytest.raises(TypeError, match="Invalid first argument"):
            traverser.register("invalid_type", dummy_func)  # type: ignore[call-overload]

    def test_register_invalid_none_with_traverser_error(self) -> None:
        """Test error when calling register(None, traverser)."""
        traverser = SingledispatchTraverser[Any]()

        def dummy_func(
            obj: Any,  # noqa: ANN401
            traverse: Any,  # noqa: ANN401
            **kwargs: Any,  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            return obj  # pragma: no cover

        with pytest.raises(TypeError, match="Invalid arguments"):
            traverser.register(None, dummy_func)  # type: ignore[call-overload]
