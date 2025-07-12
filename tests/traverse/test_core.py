"""Tests for the probly.traverse.core module."""

from __future__ import annotations

import pytest

from pytraverse import TraverserCallback
from pytraverse.core import (
    ComputedVariable,
    GlobalVariable,
    OperationNotSupportedError,
    StackVariable,
    State,
    identity_traverser,
    traverse,
    traverse_with_state,
)


class TestVariable:
    """Test base Variable functionality."""

    def test_global_variable_initialization(self) -> None:
        """Test GlobalVariable initialization and basic properties."""
        var = GlobalVariable[int]("test_var", "A test variable", 42)

        assert var.__name__ == "test_var"
        assert var.doc == "A test variable"
        assert var.default == 42
        assert var.fallback is None
        assert isinstance(var.index, int)

    def test_global_variable_with_fallback(self) -> None:
        """Test GlobalVariable with fallback to another variable."""
        fallback_var = GlobalVariable[int]("fallback", default=10)
        var = GlobalVariable[int]("test_var", default=fallback_var)

        assert var.fallback is fallback_var
        assert var.default is None

    def test_stack_variable_initialization(self) -> None:
        """Test StackVariable initialization and basic properties."""
        var = StackVariable[str]("stack_var", "A stack variable", "default")

        assert var.__name__ == "stack_var"
        assert var.doc == "A stack variable"
        assert var.default == "default"
        assert var.fallback is None

    def test_computed_variable_initialization(self) -> None:
        """Test ComputedVariable initialization."""

        def compute_func(_state: State) -> str:
            return "computed"  # pragma: no cover

        var = ComputedVariable(compute_func, "computed_var", "A computed variable")

        assert var.__name__ == "computed_var"
        assert var.doc == "A computed variable"
        assert var.compute_func is compute_func

    def test_computed_variable_with_function_defaults(self) -> None:
        """Test ComputedVariable using function name and doc as defaults."""

        def my_computation(_state: State) -> int:
            """Computes something."""
            return 123  # pragma: no cover

        var = ComputedVariable(my_computation)

        assert var.__name__ == "my_computation"
        assert var.doc == "Computes something."

    def test_variable_repr(self) -> None:
        """Test variable string representation."""
        var = GlobalVariable[int]("test", default=5)
        repr_str = repr(var)

        assert "GlobalVariable" in repr_str
        assert "test" in repr_str
        assert "default=5" in repr_str


class TestState:
    """Test State class functionality."""

    def test_state_initialization_with_traverser(self) -> None:
        """Test State initialization with a traverser."""
        state: State = State(traverser=identity_traverser)

        assert state.traverser is identity_traverser
        assert state.parent is None
        assert state.object is None
        assert state.meta is None

    def test_state_initialization_with_parent(self) -> None:
        """Test State initialization with a parent state."""
        parent_state: State[str] = State(traverser=identity_traverser)
        child_state = State(parent=parent_state, obj="test_obj", meta="test_meta")

        assert child_state.parent is parent_state
        assert child_state.traverser is identity_traverser
        assert child_state.object == "test_obj"
        assert child_state.meta == "test_meta"

    def test_state_initialization_without_traverser(self) -> None:
        """Test that State raises error when no traverser provided for root state."""
        state: State = State()
        assert state.traverser is identity_traverser

    def test_push_and_pop(self) -> None:
        """Test pushing and popping states."""
        root_state: State = State(traverser=identity_traverser)

        # Push a new state
        child_state = root_state.push("object1", "meta1")
        assert child_state.parent is root_state
        assert child_state.object == "object1"
        assert child_state.meta == "meta1"

        # Pop back to parent
        popped_state = child_state.pop()
        assert popped_state is root_state

    def test_pop_root_state_error(self) -> None:
        """Test that popping from root state raises error."""
        root_state: State = State(traverser=identity_traverser)

        with pytest.raises(ValueError, match="Cannot pop from the root state"):
            root_state.pop()

    def test_get_object_and_meta(self) -> None:
        """Test getting object and metadata from state."""
        root_state: State[str] = State(traverser=identity_traverser)
        child_state = root_state.push(
            "test_object",
            {"key": "value"},
            identity_traverser,
        )

        assert child_state.get_object() == "test_object"
        assert child_state.get_meta() == {"key": "value"}

    def test_get_object_error_when_none(self) -> None:
        """Test that get_object raises error when no object is set."""
        root_state: State = State(traverser=identity_traverser)

        with pytest.raises(ValueError, match="No object associated with this state"):
            root_state.get_object()

    def test_get_path(self) -> None:
        """Test getting the full traversal path."""
        root_state: State[str] = State(obj="x", meta="y", traverser=identity_traverser)
        child1 = root_state.push("obj1", "meta1")
        child2 = child1.push("obj2", "meta2")
        child3 = child2.push("obj3", "meta3")

        path = child3.get_path()
        expected = [("x", "y"), ("obj1", "meta1"), ("obj2", "meta2"), ("obj3", "meta3")]
        assert path == expected

    def test_get_path_objects_and_metas(self) -> None:
        """Test getting path objects and metadata separately."""
        root_state: State = State(traverser=identity_traverser)
        child1 = root_state.push("obj1", "meta1")
        child2 = child1.push("obj2", "meta2")

        objects = list(child2.get_path_objects())
        metas = list(child2.get_path_metas())

        assert objects == [None, "obj1", "obj2"]
        assert metas == [None, "meta1", "meta2"]

    def test_variable_access_via_state(self) -> None:
        """Test accessing variables through state dictionary-like interface."""
        var = GlobalVariable[int]("test_var", default=10)
        state: State = State(traverser=identity_traverser)

        # Test __contains__
        assert var in state
        assert "not_a_variable" not in state

        # Test __getitem__
        assert state[var] == 10

        # Test __setitem__
        state[var] = 20
        assert state[var] == 20

    def test_update_multiple_variables(self) -> None:
        """Test updating multiple variables at once."""
        var1 = GlobalVariable[int]("var1", default=1)
        var2 = GlobalVariable[str]("var2", default="default")
        state: State = State(traverser=identity_traverser)

        updated_state = state.update({var1: 100, var2: "updated"})

        assert updated_state is state  # Should return same instance
        assert state[var1] == 100
        assert state[var2] == "updated"


class TestGlobalVariable:
    """Test GlobalVariable specific functionality."""

    def test_global_variable_persistence(self) -> None:
        """Test that global variables persist across state hierarchy."""
        var = GlobalVariable[int]("global_var", default=0)
        root_state: State = State(traverser=identity_traverser)

        # Set in root state
        root_state[var] = 42

        # Create child state and verify value persists
        child_state = root_state.push("obj", "meta")
        assert child_state[var] == 42

        # Modify in child state
        child_state[var] = 100

        # Verify change is visible in root state
        assert root_state[var] == 100

    def test_global_variable_with_fallback(self) -> None:
        """Test global variable fallback behavior."""
        fallback_var = GlobalVariable[int]("fallback", default=99)
        var = GlobalVariable[int]("main", default=fallback_var)
        state: State = State(traverser=identity_traverser)

        # Should get fallback value initially
        assert state[var] == 99

        # Set fallback value
        state[fallback_var] = 200
        assert state[var] == 200

        # Set main variable value, should override fallback
        state[var] = 300
        assert state[var] == 300


class TestStackVariable:
    """Test StackVariable specific functionality."""

    def test_stack_variable_scoping(self) -> None:
        """Test that stack variables are scoped to their state frame."""
        var = StackVariable[str]("stack_var", default="default")
        root_state: State = State(traverser=identity_traverser)

        # Set in root state
        root_state[var] = "root_value"
        assert root_state[var] == "root_value"

        # Create child state - should inherit value
        child_state = root_state.push("obj", "meta")
        assert child_state[var] == "root_value"

        # Set in child state
        child_state[var] = "child_value"
        assert child_state[var] == "child_value"

        # Root state should be unchanged
        assert root_state[var] == "root_value"

    def test_stack_variable_get_stack(self) -> None:
        """Test getting the full stack of values."""
        var = StackVariable[int]("stack_var", default=0)
        root_state: State = State(traverser=identity_traverser)

        root_state[var] = 1
        child1 = root_state.push("obj1", "meta1")
        child1[var] = 2
        child2 = child1.push("obj2", "meta2")
        child2[var] = 3

        stack = var.get_stack(child2)
        assert stack == [1, 2, 3]

    def test_stack_variable_with_fallback(self) -> None:
        """Test stack variable with fallback behavior."""
        fallback_var = StackVariable[str]("fallback", default="fallback_default")
        var = StackVariable[str]("main", default=fallback_var)
        state: State = State(traverser=identity_traverser)

        # Should get fallback default initially
        assert state[var] == "fallback_default"

        # Set fallback value
        state[fallback_var] = "fallback_set"
        assert state[var] == "fallback_set"

        # Set main variable, should override
        state[var] = "main_set"
        assert state[var] == "main_set"


class TestComputedVariable:
    """Test ComputedVariable specific functionality."""

    def test_computed_variable_get(self) -> None:
        """Test getting computed variable value."""
        global_var = GlobalVariable[int]("base_value", default=10)

        def compute_double(state: State[object]) -> int:
            return state[global_var] * 2

        computed_var = ComputedVariable(compute_double, "doubled")
        state: State = State(traverser=identity_traverser)

        assert computed_var.get(state) == 20

        # Change base value and verify computation updates
        state[global_var] = 5
        assert computed_var.get(state) == 10

    def test_computed_variable_set_raises_error(self) -> None:
        """Test that setting computed variable raises error."""

        def compute_func(_state: State) -> int:
            return 42  # pragma: no cover

        computed_var = ComputedVariable(compute_func)
        state: State = State(traverser=identity_traverser)

        with pytest.raises(
            OperationNotSupportedError,
            match="Computed variables cannot be set directly",
        ):
            computed_var.set(state, 100)

    def test_computed_variable_callable(self) -> None:
        """Test that computed variable can be called like a function."""

        def compute_func(_state: State) -> str:
            return "computed_result"

        computed_var = ComputedVariable(compute_func)
        state: State = State(traverser=identity_traverser)

        # Test both get() and __call__()
        assert computed_var.get(state) == "computed_result"
        assert computed_var(state) == "computed_result"


class TestTraversal:
    """Test traversal functionality."""

    def test_identity_traverser(self) -> None:
        """Test the identity traverser."""
        obj = {"key": "value"}
        result = traverse(obj, identity_traverser)
        assert result is obj

    def test_traverse_with_init_variables(self) -> None:
        """Test traverse with initial variable values."""
        counter_var = GlobalVariable[int]("counter", default=0)

        def counting_traverser(
            obj: str,
            state: State[str],
            traverse: TraverserCallback[str],
        ) -> tuple:
            # traverse callback is not used in this simple test
            state[counter_var] = state[counter_var] + 1
            return obj, state

        obj = "test"
        result, state = traverse_with_state(
            obj,
            counting_traverser,
            init={counter_var: 10},
        )

        assert result == "test"
        # The counter should have been incremented from the initial value
        assert state[counter_var] == 11

    def test_complex_traversal_scenario(self) -> None:
        """Test a complex traversal scenario with nested data."""
        depth_var = StackVariable[int]("depth", default=0)
        visit_count_var = GlobalVariable[int]("visits", default=0)

        def dict_traverser(
            obj: object,
            state: State[object],
            traverse: TraverserCallback[object],
        ) -> tuple:
            assert len(state.get_path()) == state[depth_var] + 1
            if not isinstance(obj, dict):
                return obj, state

            # Increment depth for this level
            state[depth_var] += 1
            # Increment global visit counter
            state[visit_count_var] += 1

            result = {}
            for key, value in obj.items():
                new_value, new_state = traverse(value, state, meta=key)
                result[key] = new_value
                state = new_state

            return result, state

        data1 = {"a": {"b": {"c": 1}}, "d": 2}
        result1, state1 = traverse_with_state(data1, dict_traverser)
        result2, state2 = traverse_with_state("a", dict_traverser)

        # Should have same structure
        assert result1 == data1
        assert result2 == "a"
        # Check the depth and visit counts
        assert state1[depth_var] == 1  # Root object is a dict and thus has a depth of 1
        assert state2[depth_var] == 0  # "a" is not a dict, so depth is 0
        assert state1[visit_count_var] == 3  # Visits: 3 dicts in total, non-dict values do not increment count
        assert state2[visit_count_var] == 0  # Strings are not counted in visits


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_variable_counter_increments(self) -> None:
        """Test that variable counters increment properly."""
        # Test that creating variables increments counters
        global_counter = State._global_counter  # noqa: SLF001
        stack_counter = State._stack_counter  # noqa: SLF001
        GlobalVariable[int]("test1", default=0)
        GlobalVariable[str]("test2", default="")
        StackVariable[float]("test3", default=0.0)
        StackVariable[bool]("test4", default=False)

        assert State._global_counter == global_counter + 2  # noqa: SLF001
        assert State._stack_counter == stack_counter + 2  # noqa: SLF001

    def test_state_with_none_values(self) -> None:
        """Test state behavior with None values."""
        root_state: State = State(traverser=identity_traverser)
        child_state = root_state.push(None, None)

        assert child_state.object is None
        assert child_state.meta is None
        assert child_state.get_meta() is None

        # get_path should handle None objects correctly
        path = child_state.get_path()
        assert path == [(None, None), (None, None)]  # None objects are filtered out

    def test_empty_path_scenarios(self) -> None:
        """Test path methods with empty paths."""
        root_state: State = State(traverser=identity_traverser)

        assert root_state.get_path() == [(None, None)]
        assert list(root_state.get_path_objects()) == [None]
        assert list(root_state.get_path_metas()) == [None]
