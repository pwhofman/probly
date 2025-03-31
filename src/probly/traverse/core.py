from abc import ABC, abstractmethod
from collections import ChainMap
from collections.abc import Callable, Iterable
from typing import Any, Protocol

type GlobalState = dict[int, Any]
type StackState = ChainMap[int, Any]
type PathComponent[T] = tuple[T, Any]
type Path[T] = list[PathComponent[T]]


## Variables


class Variable[V](ABC):
    __slots__ = ("index", "__name__", "__doc__", "default", "fallback")
    index: int
    __name__: str
    __doc__: str | None
    default: V
    fallback: "Variable[V] | None"

    def __init__(
        self,
        index: int,
        name: str | None = None,
        doc: str | None = None,
        default: V | "Variable[V]" = None,
    ):
        self.index = index
        if name is not None:
            self.__name__ = name
        self.__doc__ = doc
        if isinstance(default, Variable):
            self.default = None  # type: ignore
            self.fallback = default
        else:
            self.default = default
            self.fallback = None

    @classmethod
    def _unregistered_init(
        cls,
        index: int,
        name: str | None = None,
        doc: str | None = None,
        default: V | "Variable[V]" = None,
    ):
        var = cls.__new__(cls)
        Variable.__init__(var, index, name, doc, default)
        return var

    def __repr__(self):
        return f"<{self.__class__.__name__}#{self.index} {self.__name__} (default={self.default})>"

    def _get(self, state: "State", d: GlobalState | StackState) -> V:
        if self.fallback is not None:
            if self.index in d:
                return d[self.index]
            return self.fallback.get(state)
        else:
            return d.get(self.index, self.default)

    @abstractmethod
    def get[T](self, state: "State[T]") -> V: ...

    @abstractmethod
    def set[T](self, state: "State[T]", value: V) -> "State[T]": ...

    def __call__(self, state: "State") -> V:
        return self.get(state)


class GlobalVariable[V](Variable[V]):

    def __init__(
        self,
        name: str | None,
        doc: str | None = None,
        default: V | "Variable[V]" = None,
    ):
        super().__init__(State._global_counter, name, doc, default)
        State._global_counter += 1

    def get[T](self, state: "State[T]") -> V:
        return self._get(state, state._global_state)

    def set[T](self, state: "State[T]", value: V) -> "State[T]":
        state._global_state[self.index] = value
        return state


class StackVariable[V](Variable[V]):

    def __init__(
        self,
        name: str | None,
        doc: str | None = None,
        default: V | "Variable[V]" = None,
    ):
        super().__init__(State._stack_counter, name, doc, default)
        State._stack_counter += 1

    def get[T](self, state: "State[T]") -> V:
        return self._get(state, state._stack_state)

    def get_stack[T](self, state: "State[T]") -> list[V]:
        index = self.index
        stack_vals = []
        last_val = self.default
        for m in reversed(state._stack_state.maps[:-1]):
            if index in m:
                last_val = m[index]
            stack_vals.append(last_val)

        return stack_vals

    def set[T](self, state: "State[T]", value: V) -> "State[T]":
        state._stack_state[self.index] = value
        return state


class OperationNotSupportedError(Exception):
    pass


class ComputedVariable[T, V](Variable[V]):
    __slots__ = ("compute_func", "__name__", "__doc__")
    compute_func: "Callable[[State[T]], V]"

    def __init__(
        self,
        compute_func: "Callable[[State[T]], V]",
        name: str | None = None,
        doc: str | None = None,
    ):
        self.compute_func = compute_func
        self.__name__ = name if name is not None else compute_func.__name__
        self.__doc__ = doc if doc is not None else compute_func.__doc__

    def __repr__(self):
        return f"<ComputedVariable: {self.__doc__}>"

    def get(self, state: "State[T]") -> V:
        return self.compute_func(state)

    def set(self, state, value):
        raise OperationNotSupportedError("Computed variables cannot be set directly.")


## Traverser Logic


class State[T]:
    __slots__ = ("traverser", "_global_state", "_stack_state", "parent")
    _global_counter: int = 0
    _stack_counter: int = 1
    path: StackVariable[PathComponent[T]] = StackVariable._unregistered_init(
        0, "path", default=(None, None)
    )

    def __init__(
        self,
        traverser: "Traverser[T] | None" = None,
        parent: "State[T] | None" = None,
    ):
        self.parent = parent

        if parent is not None:
            if traverser is None:
                self.traverser: Traverser[T] = parent.traverser
            else:
                self.traverser = traverser
            self._global_state: GlobalState = parent._global_state
            self._stack_state: StackState = parent._stack_state.new_child()
        else:
            if traverser is None:
                raise ValueError("Traverser must be provided for the root state.")
            self.traverser = traverser
            self._global_state = {}
            self._stack_state = ChainMap({})

    def __getitem__[V](self, var: Variable[V]) -> V:
        return var.get(self)

    def __setitem__[V](self, var: Variable[V], value: V):
        var.set(self, value)

    def __contains__(self, var: Any) -> bool:
        return isinstance(var, Variable)

    def push(
        self, obj: T, meta: Any = None, traverser: "Traverser[T] | None" = None
    ) -> "State[T]":
        new_state = State(traverser=traverser, parent=self)
        self.path.set(new_state, (obj, meta))

        return new_state

    def pop(self) -> "State[T]":
        if self.parent is None:
            raise ValueError("Cannot pop from the root state.")
        return self.parent

    def get_object(self) -> T:
        return self.path.get(self)[0]

    def get_meta(self) -> Any:
        return self.path.get(self)[1]

    def get_path(self) -> Path[T]:
        return self.path.get_stack(self)

    def get_path_objects(self) -> Iterable[T]:
        return (obj for obj, _ in self.get_path())

    def get_path_metas(self) -> Iterable[Any]:
        return (meta for _, meta in self.get_path())

    def update(self, init: dict[Variable, Any]) -> "State[T]":
        for var, val in init.items():
            var.set(self, val)

        return self


type TraverserResult[T] = tuple[T, State[T]]


class TraverserCallback[T](Protocol):
    def __call__(
        self,
        obj: T,
        state: State[T],
        meta: Any = None,
        traverser: "Traverser[T] | None" = None,
    ) -> TraverserResult[T]: ...


class Traverser[T](Protocol):
    def __call__(
        self,
        obj: T,
        state: State[T],
        traverse: TraverserCallback[T],
    ) -> TraverserResult[T]: ...


def traverse[T](
    obj: T, traverser: Traverser[T], init: dict[Variable, Any] | None = None
) -> T:
    state = State(traverser=traverser)

    if init is not None:
        state.update(init)

    def traverser_callback(
        obj: T, state: State[T], meta: Any = None, traverser: Traverser[T] | None = None
    ) -> TraverserResult[T]:
        new_state: State[T] = state.push(obj, meta, traverser)
        new_obj, new_state = new_state.traverser(obj, new_state, traverser_callback)
        new_state = new_state.pop()
        return new_obj, new_state

    new_obj, _ = traverser_callback(obj, state)

    return new_obj


def identity_traverser[T](
    obj: T,
    state: State[T],
    traverse: TraverserCallback[T],
) -> TraverserResult[T]:
    return obj, state
