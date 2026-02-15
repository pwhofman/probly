"""Decorators for creating and configuring traverser functions.

This module provides decorators and utilities for converting regular functions
into traverser functions with various modes of operation. The main decorator
automatically detects function signatures and wraps them appropriately.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
import inspect
import logging
from typing import (
    Any,
    Concatenate,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    Unpack,
    cast,
    overload,
)
import warnings

from pytraverse.core import (
    State,
    Traverser,
    TraverserCallback,
    TraverserResult,
    Variable,
    identity_traverser,
)

logger = logging.getLogger(__name__)

type Mode = Literal[
    "auto",  # automatically detect the mode based on function signature
    "obj",  # if LooseTraverser is a function with obj only
    "state",  # if LooseTraverser is a function with state only
    "obj_state",  # if LooseTraverser is a function with obj and state
    "obj_traverse",  # if LooseTraverser is a function with obj and traverse
    "full",  # if LooseTraverser is a full function with obj, state, traverse
    "full_positional",  # if LooseTraverser is a Traverser
    "identity",  # if LooseTraverser does not take any arguments
]


class StatelessTraverserCallback[T](Protocol):
    """Protocol for callback functions used during object traversal.

    A StatelessTraverserCallback defines the interface for functions that process
    objects during traversal operations. The callback receives the current object.

    Parameters:
        obj: The current object being traversed of type T.
        meta: Optional metadata that can be passed to provide additional context
            or configuration for the callback processing.
        traverser: Optional reference to the traverser instance performing the
            traversal, allowing the callback to access traverser methods or state.

    Returns:
        T: The result of processing the current object, which
            determines how the traversal should continue or what transformations
            should be applied.
    """

    def __call__(  # noqa: D102
        self,
        obj: T,
        meta: Any = None,  # noqa: ANN401
        traverser: Traverser[T] | None = None,
    ) -> T: ...


type ObjTraverser[T] = Callable[[T], T] | Callable[Concatenate[T, ...], T]
type ObjTraverserWithVarUpdates[T] = (
    Callable[
        [T],
        tuple[T, dict[str, Any]],
    ]
    | Callable[
        Concatenate[T, ...],
        tuple[T, dict[str, Any]],
    ]
)


type StateTraverser[T] = Callable[[State[T]], State[T]]
type ObjStateTraverser[T] = Callable[
    [T, State[T]],
    TraverserResult[T],
]

type ObjTraverseTraverser[T] = (
    Callable[
        [T, StatelessTraverserCallback[T]],
        TraverserResult[T],
    ]
    | Callable[
        Concatenate[T, StatelessTraverserCallback[T], ...],
        TraverserResult[T],
    ]
)
type ObjTraverseTraverserWithVarUpdates[T] = (
    Callable[
        [T, StatelessTraverserCallback[T]],
        tuple[T, dict[str, Any]],
    ]
    | Callable[
        Concatenate[T, StatelessTraverserCallback[T], ...],
        tuple[T, dict[str, Any]],
    ]
)

type LooseTraverserWithoutVarUpdates[T] = (
    ObjTraverser[T]
    | StateTraverser[T]
    | ObjStateTraverser[T]
    | ObjTraverseTraverser[T]
    | Traverser[T]
    | Callable[[], Any]
)
type LooseTraverserWithVarUpdates[T] = ObjTraverserWithVarUpdates[T] | ObjTraverseTraverserWithVarUpdates[T]
type LooseTraverser[T] = LooseTraverserWithoutVarUpdates[T] | LooseTraverserWithVarUpdates[T]

type StatePredicate[T] = Callable[[State[T]], bool] | Variable[bool]


def _skip_if[T](traverser: Traverser[T], pred: StatePredicate[T]) -> Traverser[T]:
    """Create a conditional traverser that skips execution based on a predicate.

    Args:
        traverser: The base traverser to conditionally execute.
        pred: Predicate function or variable that determines when to skip.

    Returns:
        A new traverser that conditionally executes the base traverser.
    """

    def _traverser(
        obj: T,
        state: State[T],
        traverse: TraverserCallback[T],
    ) -> TraverserResult[T]:
        if pred(state):
            return obj, state
        return traverser(obj, state, traverse)

    return _traverser


class SignatureDetectionWarning(UserWarning):
    """Custom warning for signature detection issues in traversers."""


def _detect_traverser_type[T](  # noqa: C901, PLR0912, PLR0915
    traverser_fn: Callable,
    mode: Mode = "auto",
    ignored_args: set[str] | None = None,
) -> tuple[Mode, str | None, str | None, str | None]:
    """Detect the type and signature of a traverser function.

    Analyzes the function signature of a traverser to determine its mode of operation
    based on the parameter names and positions. The function categorizes traversers
    into different modes depending on which parameters they accept.

    Args:
        traverser_fn: The traverser function to analyze. Can be None.
        mode: The mode to use for detection.
        ignored_args: A set of argument names to ignore during detection.

    Returns:
        A tuple containing:
        - mode: The detected traverser mode, one of:
            - "identity": Function takes no arguments
            - "full_positional": Function takes obj, state, traverse in positions 0, 1, 2
            - "full": Function takes obj, state, and traverse parameters
            - "obj": Function only takes obj parameter
            - "state": Function only takes state parameter
            - "obj_state": Function takes obj and state parameters
            - "obj_traverse": Function takes obj and traverse parameters
        - obj_name: Name of the object parameter, or None if not present
        - state_name: Name of the state parameter, or None if not present
        - traverse_name: Name of the traverse parameter, or None if not present

    Raises:
        ValueError: If the function signature doesn't match any supported traverser pattern.

    Warnings:
        Logs a warning if the object parameter is not the first argument (recommended pattern).
    """
    obj_name = None
    state_name = None
    traverse_name = None
    obj_pos = None
    state_pos = None
    traverse_pos = None
    signature = inspect.signature(traverser_fn)
    arg_names = signature.parameters.keys()

    if ignored_args is not None:
        args = [(i, arg) for i, arg in enumerate(arg_names) if arg not in ignored_args]
    else:
        args = list(enumerate(arg_names))

    if not args or len(args) == 0:
        return (
            "identity",
            None,
            None,
            None,
        )

    i: int
    arg: str

    for i, arg in args:
        if (state_name is None and mode == "state") or (arg == "state" and mode == "auto"):
            state_name = arg
            state_pos = i
            continue
        if arg == "traverse" and mode == "auto":
            traverse_name = arg
            traverse_pos = i
            continue
        if (obj_name is None and mode in {"auto", "obj"}) or (arg == "obj" and mode == "auto"):
            obj_name = arg
            obj_pos = i

    if obj_pos is not None and obj_pos != 0:
        warnings.warn(
            "A traverser should always take the object as its first argument",
            SignatureDetectionWarning,
            stacklevel=2,
        )

    if obj_pos is None and state_pos is None and traverse_pos is None and len(args) >= 2:
        arg0, arg1 = args[:2]
        if mode == "obj_state":
            obj_pos, obj_name = arg0
            state_pos, state_name = arg1
        elif mode == "obj_traverse":
            obj_pos, obj_name = arg0
            traverse_pos, traverse_name = arg1
        elif mode not in {"full", "full_positional"}:
            arg_str = ", ".join([f"'{p}'" for _, p in args])
            msg = f"Traverser signature with params {arg_str} irresolvable with mode '{mode}'."
            raise ValueError(msg)
        else:
            return "full_positional", None, None, None

    if obj_name is not None and state_name is not None and traverse_name is not None:
        if obj_pos == 0 and state_pos == 1 and traverse_pos == 2:
            return "full_positional", None, None, None
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
        arg_str = ", ".join([f"'{p}'" for _, p in args])
        msg = f"Traverser signature with params {arg_str} irresolvable with mode '{mode}'."
        raise ValueError(msg)

    return mode, obj_name, state_name, traverse_name


class TraverserDecoratorKwargs[T](TypedDict):
    """Type definition for traverser decorator keyword arguments."""

    traverse_if: NotRequired[StatePredicate[T] | None]
    skip_if: NotRequired[StatePredicate[T] | None]
    vars: NotRequired[dict[str, Variable] | None]
    type: NotRequired[type[T]] = object


@overload
def traverser[T](
    # Since mypy does not fully support generic decorators,
    # we specify the LooseTraverser without the generic type here (and below).
    # This impedes the type checking accuracy, but is necessary for compatibility.
    # See https://github.com/python/mypy/issues/17621
    traverser_fn: LooseTraverserWithoutVarUpdates[T],
    *,
    mode: Literal["auto"] = "auto",
    update_vars: Literal[False] = False,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: LooseTraverserWithVarUpdates[T],
    *,
    mode: Literal["auto"] = "auto",
    update_vars: Literal[True],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: LooseTraverser[T],
    *,
    mode: Literal["auto"] = "auto",
    update_vars: bool,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: LooseTraverser[T],
    *,
    mode: Literal["identity"],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: ObjTraverser[T],
    *,
    mode: Literal["obj"],
    update_vars: Literal[False] = False,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: ObjTraverserWithVarUpdates[T],
    *,
    mode: Literal["obj"],
    update_vars: Literal[True],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: ObjTraverser[T] | ObjTraverserWithVarUpdates[T],
    *,
    mode: Literal["obj"],
    update_vars: bool,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: StateTraverser[T],
    *,
    mode: Literal["state"],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: ObjStateTraverser[T],
    *,
    mode: Literal["obj_state"],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: ObjTraverseTraverser[T],
    *,
    mode: Literal["obj_traverse"],
    update_vars: Literal[False] = False,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: ObjTraverseTraverserWithVarUpdates[T],
    *,
    mode: Literal["obj_traverse"],
    update_vars: Literal[True],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: ObjTraverseTraverser[T] | ObjTraverseTraverserWithVarUpdates[T],
    *,
    mode: Literal["obj_traverse"],
    update_vars: bool,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    traverser_fn: Traverser[T],
    *,
    mode: Literal["full", "full_positional"],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Traverser[T]: ...


@overload
def traverser[T](
    *,
    mode: Literal["auto"] = "auto",
    update_vars: Literal[False] = False,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[LooseTraverserWithoutVarUpdates[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["auto"] = "auto",
    update_vars: Literal[True],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[LooseTraverserWithVarUpdates[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["auto"] = "auto",
    update_vars: bool,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[LooseTraverser[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["identity"],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[LooseTraverser[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["obj"],
    update_vars: Literal[False] = False,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[ObjTraverser[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["obj"],
    update_vars: Literal[True],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[ObjTraverserWithVarUpdates[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["obj"],
    update_vars: bool,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[ObjTraverser[T] | ObjTraverserWithVarUpdates[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["state"],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[StateTraverser[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["obj_state"],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[ObjStateTraverser[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["obj_traverse"],
    update_vars: Literal[False] = False,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[ObjTraverseTraverser[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["obj_traverse"],
    update_vars: Literal[True],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[ObjTraverseTraverserWithVarUpdates[T]], Traverser[T]]: ...


@overload
def traverser[T](
    *,
    mode: Literal["obj_traverse"],
    update_vars: bool,
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[
    [ObjTraverseTraverser[T] | ObjTraverseTraverserWithVarUpdates[T]],
    Traverser[T],
]: ...


@overload
def traverser[T](
    *,
    mode: Literal["full", "full_positional"],
    **kwargs: Unpack[TraverserDecoratorKwargs[T]],
) -> Callable[[Traverser[T]], Traverser[T]]: ...


def traverser[T](  # noqa: C901, PLR0912, PLR0915
    traverser_fn: LooseTraverser | None = None,
    *,
    mode: Mode = "auto",
    traverse_if: StatePredicate[T] | None = None,
    skip_if: StatePredicate[T] | None = None,
    vars: dict[str, Variable] | None = None,  # noqa: A002
    update_vars: bool = False,
    type: type[T] = object,  # type: ignore[assignment]  # noqa: A002, ARG001
) -> Traverser[T] | Callable[[LooseTraverser[T]], Traverser[T]]:
    """Decorator to convert functions into proper traverser functions.

    This decorator automatically detects the signature of the input function
    and wraps it to conform to the Traverser protocol. It supports multiple
    modes of operation and can inject variables from the traversal state.

    Args:
        traverser_fn: The function to convert into a traverser. If None, returns a decorator.
        mode: The wrapping mode.
            - "auto": Automatically detect the mode based on parameter names.
              Expects parameter names `obj`, `state` and `traverse`. If no parameter named `obj`
              is present, the first parameter not called `state` or `traverse` is assumed to be `obj`.
            - "full": Function either takes three parameters named `obj`, `state` and `traverse` in arbitrary order,
              or three differently named parameters representing obj, state and traverse respectively.
            - "obj": Function only takes a single obj parameter.
            - "state": Function only takes a single state parameter.
            - "obj_state": Function takes an obj and a state parameter.
            - "obj_traverse": Function takes obj and traverse parameters.
        traverse_if: Predicate to determine when traversal should happen.
        skip_if: Predicate to determine when traversal should be skipped.
        vars: Dictionary mapping parameter names to Variables for injection.
        update_vars: Whether the function returns updated variable values.
        type: The type of the traverser function. Optional, used only for type checking.

    Returns:
        A proper Traverser function.

    Raises:
        ValueError: If incompatible options are specified.

    Example:
        >>> @traverser
        ... def my_traverser(obj, traverse):
        ...     # Process obj and call traverse on children
        ...     return processed_obj
        >>>
        >>> @traverser(vars={"depth": depth_var})
        ... def depth_aware(obj, traverse, depth):
        ...     # Use depth variable from state
        ...     return process_with_depth(obj, depth)
    """
    # Use as decorator:
    if traverser_fn is None:

        def _decorator(fn: LooseTraverser[T]) -> Traverser[T]:
            return traverser(
                fn,
                mode=mode,  # type: ignore[arg-type]
                traverse_if=traverse_if,
                skip_if=skip_if,
                vars=vars,
                update_vars=update_vars,
            )

        return _decorator

    if vars is None and update_vars:
        msg = "Cannot use `update_vars=True` without `vars`."
        raise ValueError(msg)
    if vars is not None and mode in {"obj_state", "state", "full"}:
        msg = f"Cannot use both `vars` and `mode='{mode}'` at the same time."
        raise ValueError(msg)

    # Directly wrap traverser_fn:
    detected_mode, obj_name, state_name, traverse_name = _detect_traverser_type(
        traverser_fn,
        mode=mode,
        ignored_args=(set(vars.keys()) if vars is not None else None),
    )
    if mode == "auto":
        mode = detected_mode

    if mode == "identity":
        return identity_traverser

    if mode not in {"full", "mode_positional"} and detected_mode != "full_positional":
        if obj_name is None and mode in {"obj", "obj_state", "obj_traverse", "full"}:
            warnings.warn(
                "No positional object argument found in traverser. Using 'obj' kwarg as default.",
                SignatureDetectionWarning,
                stacklevel=2,
            )
            obj_name = "obj"
        if state_name is None and mode in {"state", "obj_state", "full"}:
            warnings.warn(
                "No positional state argument found in traverser. Using 'state' kwarg as default.",
                SignatureDetectionWarning,
                stacklevel=2,
            )
            state_name = "state"
        if traverse_name is None and mode in {"obj_traverse", "full"}:
            warnings.warn(
                "No positional traverse argument found in traverser. Using 'traverse' kwarg as default.",
                SignatureDetectionWarning,
                stacklevel=2,
            )
            traverse_name = "traverse"

    _traverser: Traverser[T]

    if mode == "full_positional":
        _traverser = traverser_fn  # type: ignore[assignment]

    elif mode == "full":
        traverser_fn = cast("Traverser[T]", traverser_fn)

        if detected_mode == "full_positional":
            # If the function is already a full positional traverser, return it directly
            return traverser_fn

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],
        ) -> TraverserResult[T]:
            return traverser_fn(  # type: ignore[call-arg]
                **{
                    obj_name: obj,
                    state_name: state,
                    traverse_name: traverse,
                },
            )

    elif mode == "obj":

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],  # noqa: ARG001
        ) -> TraverserResult[T]:
            kwargs: dict[str, Any] = {obj_name: obj}  # type: ignore[dict-item]
            if vars is not None:
                for k, v in vars.items():
                    kwargs[k] = v.get(state)
            res = traverser_fn(**kwargs)  # type: ignore[call-arg]
            if update_vars:
                obj, updates = res
                for k, v in updates.items():  # type: ignore[union-attr]
                    if k not in vars:  # type: ignore[operator]
                        continue
                    state = vars[k].set(state, v)  # type: ignore[index]
            else:
                obj = res  # type: ignore[assignment]
            return obj, state

    elif mode == "state":

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],  # noqa: ARG001
        ) -> TraverserResult[T]:
            state = traverser_fn(**{state_name: state})  # type: ignore[assignment, call-arg]
            return obj, state

    elif mode == "obj_state":

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],  # noqa: ARG001
        ) -> TraverserResult[T]:
            return traverser_fn(**{obj_name: obj, state_name: state})  # type: ignore[call-arg, return-value]

    elif mode == "obj_traverse":

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],
        ) -> TraverserResult[T]:
            def _traverse(
                obj: T,
                meta: Any = None,  # noqa: ANN401
                traverser: Traverser[T] | None = None,
            ) -> T:
                return traverse(obj, state, meta, traverser)[0]

            kwargs: dict[str, Any] = {obj_name: obj, traverse_name: _traverse}  # type: ignore[dict-item]
            if vars is not None:
                for k, v in vars.items():
                    kwargs[k] = v.get(state)
            res = traverser_fn(**kwargs)  # type: ignore[call-arg]
            if update_vars:
                obj, updates = res
                for k, v in updates.items():  # type: ignore[union-attr]
                    if k not in vars:  # type: ignore[operator]
                        continue
                    state = vars[k].set(state, v)  # type: ignore[index]
            else:
                obj = res  # type: ignore[assignment]
            return obj, state

    else:
        msg = f"Mode '{mode}' could not be applied to given traverser."
        raise ValueError(msg)

    if skip_if is not None:
        _traverser = _skip_if(_traverser, skip_if)
    if traverse_if is not None:
        _traverser = _skip_if(_traverser, lambda state: not traverse_if(state))

    if _traverser is not traverser_fn:
        _traverser = update_wrapper(_traverser, traverser_fn)

    return _traverser
