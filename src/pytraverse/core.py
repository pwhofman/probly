"""Core traversal components for the probly library.

This module provides a framework for traversing and transforming arbitrary nested structures
while maintaining state throughout the traversal. It defines variables with different scopes,
a state management system, and the traversal protocol itself.

The main components are:

- Variables: Global, Stack, and Computed variables for storing state during traversal
- State: Manages variables during traversal, with stack frames for nested structures
- Traversers: Protocol for implementing traversal logic for different data structures

Example:
```python
# Define a simple traverser for dictionaries
def dict_traverser(obj, state, traverse):
    if not isinstance(obj, dict):

    result = {}
    for k, v in obj.items():
        new_v, new_state = traverse(v, state, meta=k)
        result[k] = new_v
    return result, state

# Use the traverser
data = {"a": 1, "b": {"c": 2}}
result = traverse(data, dict_traverser)
```

OperationNotSupportedError: When attempting operations not supported by a variable type.
ValueError: When creating a root state without specifying a traverser.
ValueError: When attempting to pop from the root state.

Transformed data structure after traversal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import ChainMap
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

type GlobalState = dict[int, Any]
type StackState = ChainMap[int, Any]
type PathComponent[T] = tuple[T, Any]
type Path[T] = list[PathComponent[T]]


## Variables


class Variable[V](ABC):
    """A generic variable class for managing stateful values with fallback support.

    This abstract base class represents a variable that can be stored and retrieved from
    different state containers. Variables have an index for efficient storage/retrieval,
    optional naming and documentation, and support for default values and fallback chains.

    Type Parameters:
        V: The type of value this variable holds.

    Attributes:
        index (int): Unique identifier for efficient state storage and retrieval.
        __name__ (str): Human-readable name of the variable.
        __doc__ (str | None): Optional documentation string for the variable.
        default (V): Default value returned when variable is not set in state.
        fallback (Variable[V] | None): Optional fallback variable to query when this
            variable is not found in state.

    Abstract Methods:
        get: Retrieve the variable's value from a given state.
        set: Create a new state with this variable set to a specific value.

    Examples:
        Variables are typically used within state management systems where values
        need to be efficiently stored, retrieved, and have sensible defaults.
    """

    __slots__ = ("__name__", "default", "doc", "fallback", "index")
    index: int
    __name__: str
    doc: str | None
    default: V
    fallback: Variable[V] | None

    def __init__(
        self,
        index: int,
        name: str | None = None,
        doc: str | None = None,
        default: V | Variable[V] = None,  # type: ignore[assignment]
    ) -> None:
        """Initialize a Variable instance.

        Args:
            index: The index position of this variable.
            name: Optional name for the variable. If provided, sets the __name__ attribute.
            doc: Optional documentation string for the variable.
            default: Default value for the variable. Can be a direct value of type V or
                     another Variable instance. If a Variable is provided, it becomes the
                     fallback and default is set to None.

        Returns:
            None
        """
        self.index = index
        if name is not None:
            self.__name__ = name
        self.doc = doc
        if isinstance(default, Variable):
            self.default = None  # type: ignore[assignment]
            self.fallback = default
        else:
            self.default = default
            self.fallback = None

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Returns:
            str: A formatted string containing the class name, index, name, and default value
                in the format "<ClassName#index name (default=value)>".
        """
        return f"<{self.__class__.__name__}#{self.index} {self.__name__} (default={self.default})>"

    def _get[T](self, state: State[T], d: GlobalState | StackState) -> V:
        if self.fallback is not None:
            if self.index in d:
                return d[self.index]  # type: ignore[no-any-return]
            return self.fallback.get(state)
        return d.get(self.index, self.default)

    @abstractmethod
    def get[T](self, state: State[T]) -> V:
        """Retrieve the value of the variable in the given state.

        Args:
            state (State): The state object.

        Returns:
            V: The value of the variable associated with the provided state.
        """
        ...

    @abstractmethod
    def set[T](self, state: State[T], value: V) -> State[T]:
        """Set the variable value for the given state.

        Args:
            state: The state object to modify
            value: The value to set at the traversal location

        Returns:
            A new state object with the value set at the specified location

        Type Parameters:
            T: The type of the state object
        """
        ...

    def __call__[T](self, state: State[T]) -> V:
        """Variables are callable. They delegate calls to the get method.

        Args:
            state (State): The state object to retrieve a value for.

        Returns:
            V: The value associated with the given state.
        """
        return self.get(state)


class GlobalVariable[V](Variable[V]):
    """A variable that maintains global state across all traversal states.

    GlobalVariable extends Variable to provide a mechanism for storing and accessing
    values that persist globally across different state instances.

    The variable is automatically assigned a unique global index during initialization
    and increments the global counter to ensure uniqueness across all global variables.

    Example:
        >>> counter = GlobalVariable[int]("counter", "Global step counter", 0)
        >>> state = State()
        >>> current_count = counter.get(state)
        >>> new_state = counter.set(state, current_count + 1)
    """

    def __init__(
        self,
        name: str | None,
        doc: str | None = None,
        default: V | Variable[V] = None,  # type: ignore[assignment]
    ) -> None:
        """Initialize a new State instance.

        Args:
            name: Optional name identifier for the state variable.
            doc: Optional documentation string describing the state variable.
            default: Optional default value for the state variable, can be a direct value
                of type V or a Variable containing a value of type V.

        Returns:
            None
        """
        super().__init__(State._global_counter, name, doc, default)  # noqa: SLF001
        State._global_counter += 1  # noqa: SLF001

    def get[T](self, state: State[T]) -> V:
        """Get the value associated with the given state.

        Args:
            state: The state object to retrieve the value for.

        Returns:
            The value of type V associated with the provided state.
        """
        return self._get(state, state._global_state)  # noqa: SLF001

    def set[T](self, state: State[T], value: V) -> State[T]:
        """Set a value in the global state at this traversal's index position.

        Args:
            state: The state object to modify
            value: The value to store at this traversal's index position

        Returns:
            The modified state object

        Note:
            This method directly modifies the state's global state storage by
            setting the value at the pre-assigned index for this traversal.
        """
        state._global_state[self.index] = value  # noqa: SLF001
        return state


class StackVariable[V](Variable[V]):
    """A variable that maintains a stack of values during state traversal.

    StackVariable extends Variable to provide stack-based value storage, where each
    state frame can have its own value while maintaining access to the complete
    stack history. This is useful for tracking context that changes as you traverse
    through different states but where you need access to previous values.

    The variable automatically assigns itself a unique index from the global stack
    counter and provides methods to get the current value, retrieve the entire
    stack history, and set values at specific stack positions.

    Type Parameters:
        V: The type of values stored in this stack variable.

    Example:
        >>> depth_var = StackVariable[int]("depth", default=0)
        >>> state = State()
        >>> new_state = depth_var.set(state, 1)
        >>> current_depth = depth_var.get(new_state)
        >>> depth_history = depth_var.get_stack(new_state)
    """

    def __init__(
        self,
        name: str | None,
        doc: str | None = None,
        default: V | Variable[V] = None,  # type: ignore[assignment]
    ) -> None:
        """Initialize a new State instance.

        Args:
            name: Optional name identifier for the variable.
            doc: Optional documentation string describing the variable.
            default: Default value for the variable, can be a direct value of type V
                or a Variable containing a value of type V.

        Returns:
            None
        """
        super().__init__(State._stack_counter, name, doc, default)  # noqa: SLF001
        State._stack_counter += 1  # noqa: SLF001

    def get[T](self, state: State[T]) -> V:
        """Retrieve the value associated with the given state.

        Args:
            state (State): The state object to retrieve the value for.

        Returns:
            V: The value associated with the provided state.
        """
        return self._get(state, state._stack_state)  # noqa: SLF001

    def get_stack[T](self, state: State[T]) -> list[V]:
        """Retrieve the stack of values for this variable from the given state.

        Args:
            state: The state object to traverse.

        Returns:
            A list of values of type V, representing the stack of values for this
            variable, starting with the root state, going down to the passed state.
        """
        index = self.index
        stack_vals = []
        last_val = self.default
        for m in reversed(state._stack_state.maps):  # noqa: SLF001
            if index in m:
                last_val = m[index]
            stack_vals.append(last_val)

        return stack_vals

    def set[T](self, state: State[T], value: V) -> State[T]:
        """Set a scoped value for the passed state.

        Args:
            state: The state object representing a stack frame / traversal position.
            value: The value to set at the stack position.

        Returns:
            The modified state object with the value set for this variable.
        """
        state._stack_state[self.index] = value  # noqa: SLF001
        return state


class OperationNotSupportedError(Exception):
    """Exception raised when an operation is not supported.

    This exception is typically raised when a method is called on an object that
    does not support the requested operation, or when a specific functionality
    is not implemented for a particular operation.

    Attributes:
        None

    Inherits from:
        Exception
    """


class ComputedVariable[T, V](Variable[V]):
    """A variable whose value is computed dynamically from the current state.

    ComputedVariable represents a read-only variable that derives its value by applying
    a computation function to the current state. Unlike regular variables, computed
    variables cannot be set directly and will raise an error if modification is attempted.

    Type Parameters:
        T: The type of the state object
        V: The type of the computed value

    Raises:
        OperationNotSupportedError: When attempting to set the variable's value

    Example:
        >>> def compute_total(state):
        ...     return state.get('price') * state.get('quantity')
        >>> total_var = ComputedVariable(compute_total, name='total')
        >>> value = total_var.get(state)  # Returns computed value
        >>> total_var.set(state, 100)     # Raises OperationNotSupportedError
    """

    __slots__ = ("__name__", "compute_func", "doc")
    compute_func: Callable[[State[T]], V]

    def __init__(
        self,
        compute_func: Callable[[State[T]], V],
        name: str | None = None,
        doc: str | None = None,
    ) -> None:
        """Initialize a new instance.

        Args:
            compute_func: A callable that takes a State[T] and returns a value of type V.
            name: Optional name for the instance. If None, uses the compute_func's __name__.
            doc: Optional documentation string. If None, uses the compute_func's __doc__.
        """
        self.compute_func = compute_func
        self.__name__ = name if name is not None else compute_func.__name__
        self.doc = doc if doc is not None else compute_func.__doc__

    def __repr__(self) -> str:
        """Return a string representation of the ComputedVariable instance.

        Returns:
            str: A formatted string containing the class name and the instance's docstring,
                 in the format "<ComputedVariable: {docstring}>".
        """
        return f"<ComputedVariable: {self.__doc__}>"

    def get[Q](self, state: State[Q]) -> V:
        """Compute and return the value for the given state.

        Args:
            state (State): The state for which to compute the value.

        Returns:
            V: The computed value for the given state.
        """
        return self.compute_func(state)  # type: ignore[arg-type]

    def set[Q](self, state: State[Q], value: V) -> State[Q]:  # noqa: ARG002
        """Set the value of a computed variable in the given state.

        Args:
            state: The current state object
            value: The value to set for the computed variable

        Returns:
            State[Q]: The updated state object

        Raises:
            OperationNotSupportedError: Always raised since computed variables
                cannot be set directly and must be derived from other variables
        """
        msg = "Computed variables cannot be set directly."
        raise OperationNotSupportedError(msg)


## Traverser Logic


class State[T]:
    """A traversal state that manages context during tree/graph traversal operations.

    The State class provides a hierarchical context management system for traversing
    data structures. It maintains both global and stack-based state variables, with
    support for pushing/popping contexts as traversal progresses through nested
    structures.

    Type Parameters:
        T: The type of objects being traversed.

    Attributes:
        parent: The parent state in the traversal hierarchy, or None for root state.
        traverser: The traverser instance responsible for the traversal logic.
        object: The object associated with this state.
        meta: The metadata associated with this state's object.

    Class Attributes:
        _global_counter: Counter for global variable indices.
        _stack_counter: Counter for stack variable indices.

    The state supports variable access through dictionary-like syntax and provides
    methods for navigating the traversal hierarchy and accessing path information.

    Example:
        >>> state = State(traverser=my_traverser)
        >>> child_state = state.push(some_object, metadata)
        >>> obj = child_state.get_object()
        >>> path = child_state.get_path()
    """

    __slots__ = (
        "_global_state",
        "_stack_state",
        "meta",
        "object",
        "parent",
        "traverser",
    )
    _global_counter: int = 0
    _stack_counter: int = 0

    def __init__(
        self,
        obj: T | None = None,
        meta: Any = None,  # noqa: ANN401
        traverser: Traverser[T] | None = None,
        parent: State[T] | None = None,
    ) -> None:
        """Initialize a State instance.

        Args:
            traverser: The traverser instance to use for this state. If None and parent
                is provided, inherits the traverser from the parent state.
            parent: The parent state instance. If None, this becomes a root state.
            obj: The object associated with this state. Defaults to None for root state.
            meta: The metadata associated with the object. Defaults to None.

        Raises:
            ValueError: If traverser is None and parent is None (root state requires
                a traverser).

        Note:
            When a parent is provided:
            - The traverser is inherited from parent if not explicitly provided
            - Global state is shared with the parent
            - Stack state creates a new child scope from the parent's stack state

            When no parent is provided (root state):
            - If no traverser is provided, the identity traverser is used
            - New global and stack states are initialized
        """
        self.object = obj
        self.meta = meta
        self.parent = parent

        if parent is not None:
            if traverser is None:
                self.traverser: Traverser[T] = parent.traverser
            else:
                self.traverser = traverser
            self._global_state: GlobalState = parent._global_state  # noqa: SLF001
            self._stack_state: StackState = (
                parent._stack_state.new_child()  # noqa: SLF001
            )
        else:
            if traverser is None:
                traverser = identity_traverser
            self.traverser = traverser
            self._global_state = {}
            self._stack_state = ChainMap({})

    def __getitem__[V](self, var: Variable[V]) -> V:
        """Retrieve the value of the specified variable from the state.

        Args:
            var: The variable to retrieve.

        Returns:
            The value associated with the given variable.
        """
        return var.get(self)

    def __setitem__[V](self, var: Variable[V], value: V) -> None:
        """Set the value of the specified variable in the current state.

        Args:
            var: The variable to set.
            value: The new value associated with the variable.
        """
        var.set(self, value)

    def __contains__(self, var: Any) -> bool:  # noqa: ANN401
        """Check if the given variable is managed by this state.

        Args:
            var: The variable to check.

        Returns:
            True if the variable is an instance of Variable, False otherwise.
        """
        return isinstance(var, Variable)

    def push(
        self,
        obj: T,
        meta: Any = None,  # noqa: ANN401
        traverser: Traverser[T] | None = None,
    ) -> State[T]:
        """Create a new child state with the specified object and optional metadata.

        Args:
            obj: The object to push onto the state.
            meta: Supplemental metadata for the object. Defaults to None.
            traverser: Traverser for the new state. If None, inherits from the current state.

        Returns:
            The new child state containing the pushed object.
        """
        return State(obj=obj, meta=meta, traverser=traverser, parent=self)

    def pop(self) -> State[T]:
        """Pop the current state and return its parent.

        Raises:
            ValueError: If attempting to pop the root state (no parent).

        Returns:
            The parent state, which becomes the new current state.
        """
        if self.parent is None:
            msg = "Cannot pop from the root state."
            raise ValueError(msg)
        return self.parent

    def get_object(self) -> T:
        """Retrieve the object currently associated with this state.

        Returns:
            The object associated with this state.

        Raises:
            ValueError: If no object is associated with this state.
        """
        if self.object is None:
            msg = "No object associated with this state."
            raise ValueError(msg)
        return self.object

    def get_meta(self) -> Any:  # noqa: ANN401
        """Retrieve the metadata associated with the current object.

        Returns:
            The metadata corresponding to this state's object.
        """
        return self.meta

    def get_path(self) -> Path[T]:
        """Retrieve the entire traversal path of (object, meta) pairs from the root state to this state.

        Returns:
            A list of (object, meta) pairs representing the path.
        """
        path: Path[T] = []
        current: State[T] | None = self
        while current is not None:
            path.append((current.object, current.meta))  # type: ignore[arg-type]
            current = current.parent
        return list(reversed(path))

    def get_path_objects(self) -> Iterable[T]:
        """Return an iterator of all objects in the current traversal path.

        Returns:
            An iterator of objects, from the earliest in the path to the latest.
        """
        return (obj for obj, _ in self.get_path())

    def get_path_metas(self) -> Iterable[Any]:
        """Return an iterator of all metadata entries in the current traversal path.

        Returns:
            An iterator of metadata values, from the earliest in the path to the latest.
        """
        return (meta for _, meta in self.get_path())

    def update(self, init: dict[Variable, Any]) -> State[T]:
        """Update multiple variables in the current state at once.

        Args:
            init: A dictionary mapping variables to their new values.

        Returns:
            This state instance, updated with the provided variable values.
        """
        for var, val in init.items():
            var.set(self, val)

        return self


type TraverserResult[T] = tuple[T, State[T]]


class TraverserCallback[T](Protocol):
    """Protocol for callback functions used during object traversal.

    A TraverserCallback defines the interface for functions that process objects
    during traversal operations. The callback receives the current object being
    traversed along with contextual information about the traversal state.

    Parameters:
        obj: The current object being traversed of type T.
        state: The current state of the traversal operation containing context
            and metadata about the traversal progress.
        meta: Optional metadata that can be passed to provide additional context
            or configuration for the callback processing.
        traverser: Optional reference to the traverser instance performing the
            traversal, allowing the callback to access traverser methods or state.

    Returns:
        TraverserResult[T]: The result of processing the current object, which
            determines how the traversal should continue or what transformations
            should be applied.

    Note:
        This is a Protocol class defining the expected signature for traverser
        callback functions. Implementations should handle the object processing
        logic and return appropriate TraverserResult instances.
    """

    def __call__(  # noqa: D102
        self,
        obj: T,
        state: State[T],
        meta: Any = None,  # noqa: ANN401
        traverser: Traverser[T] | None = None,
    ) -> TraverserResult[T]: ...


type Traverser[T] = Callable[
    [T, State[T], TraverserCallback[T]],
    TraverserResult[T],
]


def traverse_with_state[T](
    obj: T,
    traverser: Traverser[Any],
    init: dict[Variable, Any] | None = None,
) -> TraverserResult[T]:
    """Traverse an object using a specified traverser function.

    This function provides a stateful traversal mechanism that allows for complex
    object transformations while maintaining context through a state stack.

    Args:
        obj (T): The object to traverse and potentially transform.
        traverser (Traverser[T]): A traverser function that defines how to process
            the object and its components.
        init (dict[Variable, Any] | None, optional): Initial state variables to
            populate the traversal state. Defaults to None.

    Returns:
        TraverserResult[T]: The transformed object and the final state after traversal.

    Note:
        The traversal uses a callback mechanism that maintains a state stack,
        allowing the traverser to push and pop context as it processes nested
        structures within the object.
    """
    state = State(obj=obj, traverser=traverser)

    if init is not None:
        state.update(init)

    def traverser_callback(
        obj: T,
        state: State[T],
        meta: Any = None,  # noqa: ANN401
        traverser: Traverser[T] | None = None,
    ) -> TraverserResult[T]:
        new_state: State[T] = state.push(obj, meta, traverser)
        new_obj, new_state = new_state.traverser(obj, new_state, traverser_callback)
        new_state = new_state.pop()
        return new_obj, new_state

    return traverser(obj, state, traverser_callback)


def traverse[T](
    obj: T,
    traverser: Traverser[Any],
    init: dict[Variable, Any] | None = None,
) -> T:
    """Traverse an object using a specified traverser function.

    This function provides a stateful traversal mechanism that allows for complex
    object transformations while maintaining context through a state stack.

    Args:
        obj (T): The object to traverse and potentially transform.
        traverser (Traverser[T]): A traverser function that defines how to process
            the object and its components.
        init (dict[Variable, Any] | None, optional): Initial state variables to
            populate the traversal state. Defaults to None.

    Returns:
        T: The transformed object after traversal. The type matches the input object
            type T.

    Note:
        The traversal uses a callback mechanism that maintains a state stack,
        allowing the traverser to push and pop context as it processes nested
        structures within the object.
    """
    return traverse_with_state(obj, traverser, init)[0]


def identity_traverser[T](
    obj: T,
    state: State[T],
    traverse: TraverserCallback[T],  # noqa: ARG001
) -> TraverserResult[T]:
    """Identity traverser that returns the object and state unchanged.

    This is a no-op traverser that simply passes through the input object
    and state without any transformation. It can be used as a base case
    or when no traversal logic is needed.

    Args:
        obj: The object to traverse (unchanged)
        state: The current traversal state (unchanged)
        traverse: The traverser callback function (unused)

    Returns:
        A tuple containing the original object and state unchanged
    """
    return obj, state
