import inspect
import logging
from collections.abc import Callable
from functools import update_wrapper
from typing import Any, Literal, NotRequired, Protocol, TypedDict, Unpack, overload

from probly.traverse.core import (
    State,
    Traverser,
    TraverserCallback,
    TraverserResult,
    Variable,
    identity_traverser,
)

logger = logging.getLogger(__name__)


type LooseTraverser[T] = Callable  # type: ignore

type Mode = Literal[
    "auto",
    "obj",
    "state",
    "obj_state",
    "obj_traverse",
    "full",
]
type StatePredicate = Callable[[State], bool] | Variable[bool]


class VarTraverser[T](Protocol):
    def __call__(
        self,
        obj: T,
        traverse: TraverserCallback[T],
        **kwargs,
    ) -> T: ...


class TraverserDecoratorKwargs(TypedDict):
    mode: NotRequired[Mode]
    traverse_if: NotRequired[StatePredicate | None]
    skip_if: NotRequired[StatePredicate | None]
    vars: NotRequired[dict[str, Variable] | None]
    update_vars: NotRequired[bool]


def _skip_if[T](traverser: Traverser[T], pred: StatePredicate) -> Traverser[T]:
    def _traverser(
        obj: T, state: State[T], traverse: TraverserCallback[T]
    ) -> TraverserResult[T]:
        if pred(state):
            return obj, state
        return traverser(obj, state, traverse)

    return _traverser


@overload
def traverser[T](
    **kwargs: Unpack[TraverserDecoratorKwargs],
) -> Callable[[LooseTraverser[T]], Traverser[T]]: ...


@overload
def traverser[T](
    traverser_fn: LooseTraverser[T],
    **kwargs: Unpack[TraverserDecoratorKwargs],
) -> Traverser[T]: ...


def traverser[T](  # type: ignore
    traverser_fn: LooseTraverser[T] | None = None,
    *,
    mode: Mode = "auto",
    traverse_if: StatePredicate | None = None,
    skip_if: StatePredicate | None = None,
    vars: dict[str, Variable] | None = None,
    update_vars: bool = False,
) -> Traverser[T]:
    if traverser_fn is None:

        def _decorator(fn: LooseTraverser[T]) -> Traverser[T]:
            return traverser(fn, mode=mode, traverse_if=traverse_if, skip_if=skip_if, vars=vars, update_vars=update_vars)  # type: ignore

        return _decorator  # type: ignore

    if mode == "auto":
        obj_name = None
        state_name = None
        traverse_name = None
        obj_pos = None
        state_pos = None
        traverse_pos = None
        argspec = inspect.getfullargspec(traverser_fn)

        if argspec.args:
            if len(argspec.args) == 0:
                return identity_traverser

            for i, arg in enumerate(argspec.args):
                if arg == "state":
                    state_name = arg
                    state_pos = i
                    continue
                if arg == "traverse":
                    traverse_name = arg
                    traverse_pos = i
                    continue
                if arg == "obj" or obj_name is None:
                    obj_name = arg
                    obj_pos = i

        if obj_pos is not None and obj_pos != 0:
            logger.warning(
                f"A traverser should always take the object as its first argument (is at pos {obj_pos+1})."  # type: ignore
            )

        if (
            obj_name is not None
            and state_name is not None
            and traverse_name is not None
        ):
            if obj_pos == 0 and state_pos == 1 and traverse_pos == 2:
                return traverser_fn  # type: ignore
            mode = "full"
        elif obj_name is not None and state_name is None and traverse_name is None:
            mode = "obj"
        elif obj_name is None and state_name is not None and traverse_name is None:
            mode = "state"
        elif obj_name is not None and state_name is not None and traverse_name is None:
            mode = "obj_state"
        elif obj_name is not None and state_name is None and traverse_name is not None:
            mode = "obj_traverse"

    else:
        obj_name = "obj"
        state_name = "state"
        traverse_name = "traverse"

    if vars is None and update_vars:
        raise ValueError("Cannot use `update_vars=True` without `vars`.")

    if mode == "full":
        if vars is not None:
            raise ValueError(
                "Cannot use both `vars` and `mode='full'` at the same time."
            )

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],
        ) -> TraverserResult[T]:
            return traverser_fn(
                **{obj_name: obj, state_name: state, traverse_name: traverse}  # type: ignore
            )

    elif mode == "obj":

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],
        ) -> TraverserResult[T]:
            kwargs: dict[str, Any] = {obj_name: obj}  # type: ignore
            if vars is not None:
                for k, v in vars.items():
                    kwargs[k] = v.get(state)
            res = traverser_fn(**kwargs)  # type: ignore
            if update_vars:
                obj, updates = res  # type: ignore
                for k, v in updates.items():  # type: ignore
                    if k not in vars:
                        continue
                    state = vars[k].set(state, v)  # type: ignore
            else:
                obj = res  # type: ignore
            return obj, state

    elif mode == "state":
        if vars is not None:
            raise ValueError(
                "Cannot use both `vars` and `mode='state'` at the same time."
            )

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],
        ) -> TraverserResult[T]:
            state = traverser_fn(**{state_name: state})  # type: ignore
            return obj, state

    elif mode == "obj_state":
        if vars is not None:
            raise ValueError(
                "Cannot use both `vars` and `mode='obj_state'` at the same time."
            )

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],
        ) -> TraverserResult[T]:
            return traverser_fn(**{obj_name: obj, state_name: state})  # type: ignore

    elif mode == "obj_traverse":

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],
        ) -> TraverserResult[T]:
            def _traverse(
                obj: T,
                meta: Any = None,
                traverser: "Traverser[T] | None" = None,
            ) -> TraverserResult[T]:
                return traverse(obj, state, meta, traverser)

            kwargs: dict[str, Any] = {obj_name: obj, traverse_name: _traverse}  # type: ignore
            if vars is not None:
                for k, v in vars.items():
                    kwargs[k] = v.get(state)
            res = traverser_fn(**kwargs)  # type: ignore
            if update_vars:
                obj, updates = res  # type: ignore
                for k, v in updates.items():  # type: ignore
                    if k not in vars:
                        continue
                    state = vars[k].set(state, v)  # type: ignore
            else:
                obj = res  # type: ignore
            return obj, state  # type: ignore

    else:
        raise ValueError(f"Mode '{mode}' could not be applied to given traverser.")

    if skip_if is not None:
        _traverser = _skip_if(_traverser, skip_if)
    if traverse_if is not None:
        _traverser = _skip_if(_traverser, lambda state: not traverse_if(state))

    _traverser = update_wrapper(_traverser, traverser_fn)

    return _traverser
