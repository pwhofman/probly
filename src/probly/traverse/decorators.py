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
from typing import Any, Literal, NotRequired, Protocol, TypedDict, Unpack, overload
import warnings

from probly.traverse.core import (
    State,
    Traverser,
    TraverserCallback,
    TraverserResult,
    Variable,
    identity_traverser,
)

logger = logging.getLogger(__name__)


type LooseTraverser[T] = Callable

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
type StatePredicate = Callable[[State], bool] | Variable[bool]


class VarTraverser[T](Protocol):
    """Protocol for variable-based traverser functions.

    Defines the interface for traverser functions that work with variable
    injection and state management.
    """

    def __call__(
        self,
        obj: T,
        traverse: TraverserCallback[T],
        **kwargs,  # noqa: ANN003
    ) -> T:
        """Execute the traverser with variable injection.

        Args:
            obj: The object to traverse.
            traverse: The traverser callback function.
            **kwargs: Variable values injected from state.

        Returns:
            The transformed object.
        """
        ...


def _skip_if[T](traverser: Traverser[T], pred: StatePredicate) -> Traverser[T]:
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
    traverser_fn: LooseTraverser[T],
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
    argspec = inspect.getfullargspec(traverser_fn)
    args = argspec.args

    if ignored_args is not None:
        args = [(i, arg) for i, arg in enumerate(args) if arg not in ignored_args]
    else:
        args = list(enumerate(args))

    if not args or len(args) == 0:
        return (
            "identity",
            None,
            None,
            None,
        )

    for i, arg in args:
        if (state_name is None and mode == "state") or (
            arg == "state" and mode == "auto"
        ):
            state_name = arg
            state_pos = i
            continue
        if arg == "traverse" and mode == "auto":
            traverse_name = arg
            traverse_pos = i
            continue
        if (obj_name is None and mode in {"auto", "obj"}) or (
            arg == "obj" and mode == "auto"
        ):
            obj_name = arg
            obj_pos = i

    if obj_pos is not None and obj_pos != 0:
        warnings.warn(
            "A traverser should always take the object as its first argument",
            SignatureDetectionWarning,
            stacklevel=2,
        )

    if (
        obj_pos is None
        and state_pos is None
        and traverse_pos is None
        and len(args) >= 2
    ):
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


class TraverserDecoratorKwargs(TypedDict):
    """Type definition for traverser decorator keyword arguments."""

    mode: NotRequired[Mode]
    traverse_if: NotRequired[StatePredicate | None]
    skip_if: NotRequired[StatePredicate | None]
    vars: NotRequired[dict[str, Variable] | None]
    update_vars: NotRequired[bool]


@overload
def traverser[T](
    **kwargs: Unpack[TraverserDecoratorKwargs],
) -> Callable[[LooseTraverser[T]], Traverser[T]]: ...


@overload
def traverser[T](
    traverser_fn: LooseTraverser[T],
    **kwargs: Unpack[TraverserDecoratorKwargs],
) -> Traverser[T]: ...


def traverser[T](  # noqa: C901, PGH003, PLR0912, PLR0915 # type: ignore
    traverser_fn: LooseTraverser[T] | None = None,
    *,
    mode: Mode = "auto",
    traverse_if: StatePredicate | None = None,
    skip_if: StatePredicate | None = None,
    vars: dict[str, Variable] | None = None,  # noqa: A002
    update_vars: bool = False,
) -> Traverser[T]:
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
                mode=mode,
                traverse_if=traverse_if,
                skip_if=skip_if,
                vars=vars,
                update_vars=update_vars,
            )

        return _decorator  # type: ignore  # noqa: PGH003

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

    if mode == "full_positional":
        return traverser_fn

    if mode != "full" and detected_mode != "full_positional":
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

    if mode == "full":
        if detected_mode == "full_positional":
            # If the function is already a full positional traverser, return it directly
            return traverser_fn

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],
        ) -> TraverserResult[T]:
            return traverser_fn(
                **{
                    obj_name: obj,
                    state_name: state,
                    traverse_name: traverse,
                },  # type: ignore  # noqa: PGH003
            )

    elif mode == "obj":

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],  # noqa: ARG001
        ) -> TraverserResult[T]:
            kwargs: dict[str, Any] = {obj_name: obj}  # type: ignore  # noqa: PGH003
            if vars is not None:
                for k, v in vars.items():
                    kwargs[k] = v.get(state)
            res = traverser_fn(**kwargs)  # type: ignore  # noqa: PGH003
            if update_vars:
                obj, updates = res  # type: ignore  # noqa: PGH003
                for k, v in updates.items():  # type: ignore  # noqa: PGH003
                    if k not in vars:
                        continue
                    state = vars[k].set(state, v)  # type: ignore  # noqa: PGH003
            else:
                obj = res  # type: ignore  # noqa: PGH003
            return obj, state

    elif mode == "state":

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],  # noqa: ARG001
        ) -> TraverserResult[T]:
            state = traverser_fn(**{state_name: state})  # type: ignore  # noqa: PGH003
            return obj, state

    elif mode == "obj_state":

        def _traverser(
            obj: T,
            state: State[T],
            traverse: TraverserCallback[T],  # noqa: ARG001
        ) -> TraverserResult[T]:
            return traverser_fn(**{obj_name: obj, state_name: state})  # type: ignore  # noqa: PGH003

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
            ) -> TraverserResult[T]:
                return traverse(obj, state, meta, traverser)

            kwargs: dict[str, Any] = {obj_name: obj, traverse_name: _traverse}  # type: ignore  # noqa: PGH003
            if vars is not None:
                for k, v in vars.items():
                    kwargs[k] = v.get(state)
            res = traverser_fn(**kwargs)  # type: ignore  # noqa: PGH003
            if update_vars:
                obj, updates = res  # type: ignore  # noqa: PGH003
                for k, v in updates.items():  # type: ignore  # noqa: PGH003
                    if k not in vars:
                        continue
                    state = vars[k].set(state, v)  # type: ignore  # noqa: PGH003
            else:
                obj = res  # type: ignore  # noqa: PGH003
            return obj, state  # type: ignore  # noqa: PGH003

    else:
        msg = f"Mode '{mode}' could not be applied to given traverser."
        raise ValueError(msg)

    if skip_if is not None:
        _traverser = _skip_if(_traverser, skip_if)
    if traverse_if is not None:
        _traverser = _skip_if(_traverser, lambda state: not traverse_if(state))

    _traverser = update_wrapper(_traverser, traverser_fn)

    return _traverser
