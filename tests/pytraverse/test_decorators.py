"""Tests for the probly.traverse.decorators module.

This module tests the decorator functionality for creating and configuring
traverser functions with various modes of operation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from pytraverse.core import (
    GlobalVariable,
    StackVariable,
    State,
    Traverser,
    TraverserResult,
    Variable,
    identity_traverser,
)
from pytraverse.decorators import (
    SignatureDetectionWarning,
    TraverserDecoratorKwargs,
    _detect_traverser_type,
    _skip_if,
    traverser,
)

# Test Variables
TEST_GLOBAL_VAR = GlobalVariable[int]("test_global", "A test global variable", 42)
TEST_STACK_VAR = StackVariable[str]("test_stack", "A test stack variable", "default")


def dummy_traverse(
    obj: Any,  # noqa: ANN401
    state: State,
    meta: Any = None,  # noqa: ANN401
    traverser: Traverser | None = None,
) -> TraverserResult:
    """Dummy traverse function for testing."""
    return obj, state  # pragma: no cover


class TestDetectTraverserType:
    """Test the _detect_traverser_type function."""

    def test_detect_identity_traverser(self) -> None:
        """Test detection of identity traverser (no arguments)."""

        def no_args() -> int:
            return 42  # pragma: no cover

        # Identity traverser should be detected correctly now
        mode, obj_name, state_name, traverse_name = _detect_traverser_type(no_args)
        assert mode == "identity"
        assert obj_name is None
        assert state_name is None
        assert traverse_name is None

    def test_detect_obj_only_traverser(self) -> None:
        """Test detection of obj-only traverser."""

        def obj_only(obj):  # type: ignore[no-untyped-def]
            return obj  # pragma: no cover

        mode, obj_name, state_name, traverse_name = _detect_traverser_type(obj_only)
        assert mode == "obj"
        assert obj_name == "obj"
        assert state_name is None
        assert traverse_name is None

    def test_detect_obj_only_renamed_traverser(self) -> None:
        """Test detection of obj-only traverser."""

        def obj_only(item):  # type: ignore[no-untyped-def]
            return item  # pragma: no cover

        mode, obj_name, state_name, traverse_name = _detect_traverser_type(obj_only)
        assert mode == "obj"
        assert obj_name == "item"
        assert state_name is None
        assert traverse_name is None

    def test_detect_state_only_traverser(self) -> None:
        """Test detection of state-only traverser."""

        def state_only(state):  # type: ignore[no-untyped-def]
            return state  # pragma: no cover

        mode, obj_name, state_name, traverse_name = _detect_traverser_type(state_only)
        assert mode == "state"
        assert obj_name is None
        assert state_name == "state"
        assert traverse_name is None

    def test_detect_state_only_renamed_traverser(self) -> None:
        """Test detection of state-only traverser."""

        def state_only(s):  # type: ignore[no-untyped-def]
            return s  # pragma: no cover

        mode, obj_name, state_name, traverse_name = _detect_traverser_type(
            state_only,
            "state",
        )
        assert mode == "state"
        assert obj_name is None
        assert state_name == "s"
        assert traverse_name is None

    def test_detect_obj_state_traverser(self) -> None:
        """Test detection of obj-state traverser."""

        def obj_state(state, obj):  # type: ignore[no-untyped-def]
            return obj, state  # pragma: no cover

        with pytest.warns(
            SignatureDetectionWarning,
            match="should always take the object as its first argument",
        ):
            mode, obj_name, state_name, traverse_name = _detect_traverser_type(
                obj_state,
            )
        assert mode == "obj_state"
        assert obj_name == "obj"
        assert state_name == "state"
        assert traverse_name is None

    def test_detect_obj_state_renamed_traverser(self) -> None:
        """Test detection of obj-state traverser."""

        def obj_state(o, s):  # type: ignore[no-untyped-def]
            return o, s  # pragma: no cover

        mode, obj_name, state_name, traverse_name = _detect_traverser_type(
            obj_state,
            "obj_state",
        )
        assert mode == "obj_state"
        assert obj_name == "o"
        assert state_name == "s"
        assert traverse_name is None

    def test_detect_obj_traverse_traverser(self) -> None:
        """Test detection of obj-traverse traverser."""

        def obj_traverse(traverse, obj):  # type: ignore[no-untyped-def]
            return obj  # pragma: no cover

        with pytest.warns(
            SignatureDetectionWarning,
            match="should always take the object as its first argument",
        ):
            mode, obj_name, state_name, traverse_name = _detect_traverser_type(
                obj_traverse,
            )
        assert mode == "obj_traverse"
        assert obj_name == "obj"
        assert state_name is None
        assert traverse_name == "traverse"

    def test_detect_obj_traverse_renamed_traverser(self) -> None:
        """Test detection of obj-traverse traverser."""

        def obj_traverse(item, visit):  # type: ignore[no-untyped-def]
            return item  # pragma: no cover

        mode, obj_name, state_name, traverse_name = _detect_traverser_type(
            obj_traverse,
            "obj_traverse",
        )
        assert mode == "obj_traverse"
        assert obj_name == "item"
        assert state_name is None
        assert traverse_name == "visit"

    def test_detect_full_traverser(self) -> None:
        """Test detection of full traverser."""

        def full_traverser(  # type: ignore[no-untyped-def]
            traverse,
            state,
            obj,
        ):  # Shuffled args
            return obj, state  # pragma: no cover

        with pytest.warns(
            SignatureDetectionWarning,
            match="should always take the object as its first argument",
        ):
            mode, obj_name, state_name, traverse_name = _detect_traverser_type(
                full_traverser,
            )
        assert mode == "full"
        assert obj_name == "obj"
        assert state_name == "state"
        assert traverse_name == "traverse"

    def test_detect_full_renamed_traverser(self) -> None:
        """Test detection of full traverser."""

        def full_traverser(traverse, item, state):  # type: ignore[no-untyped-def] # Shuffled args
            return item, state  # pragma: no cover

        with pytest.warns(
            SignatureDetectionWarning,
            match="should always take the object as its first argument",
        ):
            mode, obj_name, state_name, traverse_name = _detect_traverser_type(
                full_traverser,
            )
        assert mode == "full"
        assert obj_name == "item"
        assert state_name == "state"
        assert traverse_name == "traverse"

    def test_detect_full_positional_traverser(self) -> None:
        """Test detection of full positional traverser."""

        def full_positional(obj, state, traverse):  # type: ignore[no-untyped-def]
            return obj, state  # pragma: no cover

        mode, obj_name, state_name, traverse_name = _detect_traverser_type(
            full_positional,
        )
        assert mode == "full_positional"
        assert obj_name is None
        assert state_name is None
        assert traverse_name is None

    def test_detect_custom_parameter_names(self) -> None:
        """Test detection with custom parameter names."""

        def custom_names(item, current_state, visitor):  # type: ignore[no-untyped-def]
            return item, current_state  # pragma: no cover

        mode, obj_name, state_name, traverse_name = _detect_traverser_type(custom_names)
        # Since only exact matches for "state" and "traverse" are recognized,
        # this will be detected as obj-only (first param becomes obj)
        assert mode == "obj"
        assert obj_name == "item"
        assert state_name is None
        assert traverse_name is None

    def test_detect_misleading_traverse_signature(self) -> None:
        """Test detection of misleading function signature."""

        def unsupported(traverse):  # type: ignore[no-untyped-def]
            return None  # pragma: no cover

        mode, obj_name, state_name, traverse_name = _detect_traverser_type(
            unsupported,
            "obj",
        )
        assert mode == "obj"
        assert obj_name == "traverse"
        assert state_name is None
        assert traverse_name is None

    def test_detect_misleading_state_signature(self) -> None:
        """Test detection of misleading function signature."""

        def unsupported(state):  # type: ignore[no-untyped-def]
            return None  # pragma: no cover

        mode, obj_name, state_name, traverse_name = _detect_traverser_type(
            unsupported,
            "obj",
        )
        assert mode == "obj"
        assert obj_name == "state"
        assert state_name is None
        assert traverse_name is None

    def test_detect_unsupported_signature(self) -> None:
        """Test detection of unsupported function signature."""

        def unsupported(traverse):  # type: ignore[no-untyped-def]
            return None  # pragma: no cover

        with pytest.raises(
            ValueError,
            match="signature with params 'traverse' irresolvable with mode 'auto'",
        ):
            _detect_traverser_type(unsupported)


class TestSkipIf:
    """Test the _skip_if conditional wrapper."""

    def test_skip_if_with_function_predicate(self) -> None:
        """Test _skip_if with a function predicate."""
        mock_traverser = Mock(return_value=("modified", Mock()))
        predicate = Mock(return_value=True)

        conditional_traverser = _skip_if(mock_traverser, predicate)

        obj = "test"
        state: State = State()
        result: Any
        new_state: State
        result, new_state = conditional_traverser(obj, state, dummy_traverse)

        # Should skip the traverser and return original obj
        assert result == obj
        assert new_state == state
        predicate.assert_called_once_with(state)
        mock_traverser.assert_not_called()

    def test_skip_if_with_function_predicate_false(self) -> None:
        """Test _skip_if when predicate returns False."""
        expected_result = ("modified", Mock())
        mock_traverser = Mock(return_value=expected_result)
        predicate = Mock(return_value=False)

        conditional_traverser = _skip_if(mock_traverser, predicate)

        obj = "test"
        state: State = State()
        result: Any
        new_state: State
        result, new_state = conditional_traverser(obj, state, dummy_traverse)

        # Should execute the traverser
        assert (result, new_state) == expected_result
        predicate.assert_called_once_with(state)
        mock_traverser.assert_called_once_with(obj, state, dummy_traverse)

    def test_skip_if_with_variable_predicate(self) -> None:
        """Test _skip_if with a Variable predicate."""
        mock_traverser = Mock(return_value=("modified", Mock()))
        var = StackVariable[bool]("skip_flag", default=True)

        conditional_traverser: Traverser[str] = _skip_if(mock_traverser, var)

        obj = "test"
        state: State = State()
        result: Any
        new_state: State
        result, new_state = conditional_traverser(obj, state, dummy_traverse)

        # Should skip the traverser since variable default is True
        assert result == obj
        assert new_state == state
        mock_traverser.assert_not_called()


class TestTraverserDecorator:
    """Test the main traverser decorator."""

    def test_decorator_without_arguments(self) -> None:
        """Test using @traverser without arguments."""

        @traverser
        def simple_traverser(obj, traverse):  # type: ignore[no-untyped-def]
            return obj * 2

        assert callable(simple_traverser)

        obj = 5
        state: State = State()
        result: Any
        new_state: State
        result, new_state = simple_traverser(obj, state, dummy_traverse)

        assert result == 10
        assert new_state == state

    def test_decorator_with_arguments(self) -> None:
        """Test using @traverser with arguments."""

        @traverser(mode="obj")
        def obj_traverser(obj):  # type: ignore[no-untyped-def]
            return obj * 3

        obj: Any = 4
        state: State = State()
        result: Any
        new_state: State
        result, new_state = obj_traverser(obj, state, dummy_traverse)

        assert result == 12
        assert new_state == state

    def test_as_function_call(self) -> None:
        """Test using traverser as a direct function call."""

        def raw_traverser(obj, traverse):  # type: ignore[no-untyped-def]
            return obj + 1

        wrapped: Traverser = traverser(raw_traverser)

        obj = 10
        state: State = State()
        result: Any
        new_state: State
        result, new_state = wrapped(obj, state, dummy_traverse)

        assert result == 11
        assert new_state == state

    def test_mode_auto_detection(self) -> None:
        """Test automatic mode detection."""

        @traverser(mode="auto")
        def auto_traverser(obj, state, traverse):  # type: ignore[no-untyped-def]
            return obj, state

        obj: Any = "test"
        state: State = State()
        result: Any
        new_state: State
        result, new_state = auto_traverser(obj, state, dummy_traverse)

        assert result == obj
        assert new_state == state

    def test_mode_identity(self) -> None:
        """Test identity mode returns identity_traverser."""

        def no_args():  # type: ignore[no-untyped-def]
            return None

        wrapped: Traverser = traverser(no_args, mode="auto")
        assert wrapped is identity_traverser

    def test_mode_full_positional(self) -> None:
        """Test full positional mode returns original function."""

        def full_pos(obj, state, traverse):  # type: ignore[no-untyped-def]
            return obj * 2, state

        wrapped: Traverser = traverser(full_pos, mode="auto")
        assert wrapped is full_pos

    def test_mode_obj(self) -> None:
        """Test obj mode."""

        @traverser(mode="obj", type=str)
        def obj_traverser(item: str) -> str:
            return item.upper()

        obj = "hello"
        state: State = State()
        result, new_state = obj_traverser(obj, state, dummy_traverse)

        assert result == "HELLO"
        assert new_state == state

    def test_mode_state(self) -> None:
        """Test state mode."""
        new_state_mock = Mock()

        @traverser(mode="state")
        def state_traverser(state):  # type: ignore[no-untyped-def]
            return new_state_mock

        obj: Any = "test"
        state: State = State()
        result: Any
        new_state: State
        result, new_state = state_traverser(obj, state, dummy_traverse)

        assert result == obj  # Object unchanged
        assert new_state == new_state_mock

    def test_mode_obj_state(self) -> None:
        """Test obj_state mode."""

        @traverser(mode="obj_state")
        def obj_state_traverser(obj, state):  # type: ignore[no-untyped-def]
            return obj * 2, state

        obj: Any = 5
        state: State = State()
        result: Any
        new_state: State
        result, new_state = obj_state_traverser(obj, state, dummy_traverse)

        assert result == 10
        assert new_state == state

    def test_mode_obj_traverse(self) -> None:
        """Test obj_traverse mode."""

        @traverser(mode="obj_traverse")
        def obj_traverse_traverser(obj, traverse):  # type: ignore[no-untyped-def]
            return obj.upper()

        obj: Any = "hello"
        state: State = State()
        result: Any
        new_state: State
        result, new_state = obj_traverse_traverser(obj, state, dummy_traverse)

        assert result == "HELLO"
        assert new_state == state

    def test_mode_full(self) -> None:
        """Test full mode with custom parameter names."""

        @traverser(mode="full")
        def full_traverser(item, current_state, visitor):  # type: ignore[no-untyped-def]
            return item * 2, current_state

        obj: Any = 3
        state: State = State()
        result: Any
        new_state: State
        result, new_state = full_traverser(obj, state, dummy_traverse)

        assert result == 6
        assert new_state == state

    def test_vars_injection(self) -> None:
        """Test variable injection from state."""

        @traverser(mode="obj", vars={"multiplier": TEST_GLOBAL_VAR})
        def var_traverser(obj, multiplier):  # type: ignore[no-untyped-def]
            return obj * multiplier

        obj: Any = 5
        state: State = State().update({TEST_GLOBAL_VAR: 3})
        result: Any
        new_state: State
        result, new_state = var_traverser(obj, state, dummy_traverse)

        assert result == 15
        assert new_state == state

    def test_vars_injection_with_update(self) -> None:
        """Test variable injection with update_vars=True."""

        @traverser(mode="obj", vars={"counter": TEST_GLOBAL_VAR}, update_vars=True)
        def updating_traverser(obj, counter):  # type: ignore[no-untyped-def]
            return obj, {"counter": counter + 1}

        obj: Any = "test"
        state: State = State().update({TEST_GLOBAL_VAR: 5})
        result: Any
        new_state: State
        result, new_state = updating_traverser(obj, state, dummy_traverse)

        assert result == obj
        assert TEST_GLOBAL_VAR.get(new_state) == 6

    def test_skip_if_predicate(self) -> None:
        """Test skip_if functionality."""
        skip_var = StackVariable[bool]("skip", default=False)

        @traverser(mode="obj", skip_if=skip_var, type=int)
        def skippable_traverser(obj: int) -> int:
            return obj * 2

        # Test when skip is False
        obj: Any = 5
        state: State = State().update({skip_var: False})
        result: Any
        new_state: State
        result, new_state = skippable_traverser(obj, state, dummy_traverse)
        assert result == 10

        # Test when skip is True
        state = State().update({skip_var: True})
        result, new_state = skippable_traverser(obj, state, dummy_traverse)
        assert result == obj  # Should be unchanged

    def test_traverse_if_predicate(self) -> None:
        """Test traverse_if functionality."""
        run_var = StackVariable[bool]("run", default=True)

        @traverser(mode="obj", traverse_if=run_var)
        def conditional_traverser(obj):  # type: ignore[no-untyped-def]
            return obj * 2

        # Test when run is True
        obj: Any = 5
        state: State = State().update({run_var: True})
        result: Any
        new_state: State
        result, new_state = conditional_traverser(obj, state, dummy_traverse)
        assert result == 10

        # Test when run is False
        state = State().update({run_var: False})
        result, new_state = conditional_traverser(obj, state, dummy_traverse)
        assert result == obj  # Should be unchanged

    def test_error_vars_without_update_vars(self) -> None:
        """Test error when update_vars=True but no vars provided."""
        with pytest.raises(
            ValueError,
            match="Cannot use `update_vars=True` without `vars`",
        ):

            @traverser(update_vars=True)
            def bad_traverser(obj):  # type: ignore[no-untyped-def]
                return obj  # pragma: no cover

    def test_error_vars_with_full_mode(self) -> None:
        """Test error when using vars with full mode."""
        with pytest.raises(
            ValueError,
            match="Cannot use both `vars` and `mode='full'`",
        ):

            @traverser(mode="full", vars={"test": TEST_GLOBAL_VAR})
            def bad_traverser(obj, state, traverse):  # type: ignore[no-untyped-def]
                return obj, state  # pragma: no cover

    def test_error_vars_with_state_mode(self) -> None:
        """Test error when using vars with state mode."""
        with pytest.raises(
            ValueError,
            match="Cannot use both `vars` and `mode='state'`",
        ):

            @traverser(mode="state", vars={"test": TEST_GLOBAL_VAR})
            def bad_traverser(state):  # type: ignore[no-untyped-def]
                return state  # pragma: no cover

    def test_error_vars_with_obj_state_mode(self) -> None:
        """Test error when using vars with obj_state mode."""
        with pytest.raises(
            ValueError,
            match="Cannot use both `vars` and `mode='obj_state'`",
        ):

            @traverser(mode="obj_state", vars={"test": TEST_GLOBAL_VAR})
            def bad_traverser(obj, state):  # type: ignore[no-untyped-def]
                return obj, state  # pragma: no cover

    def test_error_invalid_mode(self) -> None:
        """Test error when using invalid mode."""
        with pytest.raises(
            ValueError,
            match="signature with params 'obj' irresolvable with mode 'invalid'",
        ):

            @traverser(mode="invalid")  # type: ignore[call-overload]
            def bad_traverser(obj):  # type: ignore[no-untyped-def]
                return obj  # pragma: no cover

        with pytest.raises(
            ValueError,
            match="signature with params 'obj', 'state' irresolvable with mode 'invalid'",
        ):

            @traverser(mode="invalid")  # type: ignore[call-overload]
            def bad_traverser(obj, state):  # type: ignore[no-untyped-def]
                return obj  # pragma: no cover

        with pytest.raises(
            ValueError,
            match="signature with params 'obj', 'state', 'traverse' irresolvable with mode 'invalid'",
        ):

            @traverser(mode="invalid")  # type: ignore[call-overload]
            def bad_traverser(obj, state, traverse):  # type: ignore[no-untyped-def]
                return obj  # pragma: no cover

    def test_wrapper_preserves_function_metadata(self) -> None:
        """Test that the wrapper preserves original function metadata."""

        def original_function(obj, traverse):  # type: ignore[no-untyped-def]
            """Original docstring."""
            return obj  # pragma: no cover

        wrapped: Traverser = traverser(original_function)

        assert hasattr(wrapped, "__name__")
        assert hasattr(wrapped, "__doc__")
        assert wrapped.__name__ == "original_function"
        assert wrapped.__doc__ == "Original docstring."

    def test_obj_traverse_mode_with_traverse_callback(self) -> None:
        """Test obj_traverse mode with actual traverse callback usage."""
        calls = []

        def mock_traverse(obj, state, meta=None, traverser=None):  # type: ignore[no-untyped-def]
            calls.append((obj, meta))
            return obj * 2, state

        @traverser(mode="obj_traverse")
        def traverse_using_traverser(obj, traverse):  # type: ignore[no-untyped-def]
            result1 = traverse(obj)
            result2 = traverse(obj + 1, meta="child")
            return result1 + result2

        obj: Any = 5
        state: State = State()
        result: Any
        new_state: State
        result, new_state = traverse_using_traverser(obj, state, mock_traverse)

        assert result == 22  # (5*2) + (6*2)
        assert len(calls) == 2
        assert calls[0] == (5, None)
        assert calls[1] == (6, "child")


class TestTraverserDecoratorKwargs:
    """Test TraverserDecoratorKwargs TypedDict."""

    def test_decorator_kwargs_structure(self) -> None:
        """Test that TraverserDecoratorKwargs has expected structure."""
        # This is mainly a type check test
        kwargs: TraverserDecoratorKwargs = {
            "traverse_if": None,
            "skip_if": None,
            "vars": None,
        }

        assert kwargs["traverse_if"] is None
        assert kwargs["skip_if"] is None
        assert kwargs["vars"] is None

    def test_decorator_kwargs_partial(self) -> None:
        """Test TraverserDecoratorKwargs with partial specification."""
        v: dict[str, Variable] = {
            "test_var": TEST_GLOBAL_VAR,
            "another_var": TEST_STACK_VAR,
        }
        kwargs: TraverserDecoratorKwargs = {
            "vars": v,
        }

        assert kwargs["vars"] == v


class TestComplexScenarios:
    """Test complex scenarios combining multiple features."""

    def test_complex_traverser_with_all_features(self) -> None:
        """Test a complex traverser using multiple features."""
        counter_var = StackVariable[int]("counter", default=0)
        active_var = StackVariable[bool]("active", default=True)

        @traverser(
            mode="obj_traverse",
            vars={"counter": counter_var},
            update_vars=True,
            traverse_if=active_var,
        )
        def complex_traverser(obj, traverse, counter):  # type: ignore[no-untyped-def]
            # Increment counter and process object
            new_counter = counter + 1
            processed_obj = f"item_{new_counter}_{obj}"
            return processed_obj, {"counter": new_counter}

        # Test when active
        obj: Any = "test"
        state: State = State().update({counter_var: 5, active_var: True})
        result: Any
        new_state: State
        result, new_state = complex_traverser(obj, state, dummy_traverse)

        assert result == "item_6_test"
        assert counter_var.get(new_state) == 6

        # Test when not active
        state = State().update({counter_var: 5, active_var: False})
        result, new_state = complex_traverser(obj, state, dummy_traverse)

        assert result == obj  # Should be unchanged
        assert counter_var.get(new_state) == 5  # Counter should not increment

    def test_nested_conditional_traversers(self) -> None:
        """Test nested conditional traversers."""
        flag1 = StackVariable[bool]("flag1", default=False)
        flag2 = StackVariable[bool]("flag2", default=False)

        @traverser(mode="obj", skip_if=flag1, type=int)
        def first_traverser(obj):  # type: ignore[no-untyped-def]
            return obj * 2

        @traverser(mode="obj", skip_if=flag2, type=int)
        def second_traverser(obj):  # type: ignore[no-untyped-def]
            return obj + 10

        # Test both flags False
        obj: Any = 5
        state: State = State().update({flag1: False, flag2: False})

        result1, state1 = first_traverser(obj, state, dummy_traverse)
        result2, state2 = second_traverser(result1, state1, dummy_traverse)

        assert result2 == 20  # (5 * 2) + 10

        # Test first flag True
        state = State().update({flag1: True, flag2: False})

        result1, state1 = first_traverser(obj, state, dummy_traverse)
        result2, state2 = second_traverser(result1, state1, dummy_traverse)

        assert result2 == 15  # 5 + 10 (first traverser skipped)

    def test_variable_fallback_behavior(self) -> None:
        """Test behavior with variable fallbacks."""
        fallback_var = GlobalVariable[int]("fallback", default=100)
        composite_var = GlobalVariable[int]("composite", default=fallback_var)

        @traverser(mode="obj", vars={"value": composite_var}, type=int)
        def fallback_traverser(obj: int, value: int) -> int:
            return obj * value

        # Test with only fallback set
        obj: Any = 3
        state: State = State().update({fallback_var: 50})
        result: Any
        new_state: State
        result, new_state = fallback_traverser(obj, state, dummy_traverse)

        assert result == 150  # 3 * 50 (using fallback)
