"""Pickle helpers that preserve explicit RegistryMeta instance registrations."""

from __future__ import annotations

import io
import pickle
from typing import TYPE_CHECKING, Any, cast

from flextype.registry_meta import RegistryMeta, iter_registry_classes

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    type _StateSetter = Callable[[object, object], Any]


_REGISTRY_STATE_MARKER = "flextype.registry_pickle.state.v1"


def _collect_explicit_registry_classes(instance: object) -> list[RegistryMeta[Any]]:
    """Collect all registry classes where `instance` was explicitly registered."""
    return [
        registry_class
        for registry_class in iter_registry_classes()
        if registry_class.is_explicit_instance_registered(instance)
    ]


def _normalize_reduce(
    reduced: object,
) -> tuple[Callable[..., Any], tuple[Any, ...], object, object, object, _StateSetter | None] | None:
    """Normalize a reduce result into a 6-item tuple."""
    if isinstance(reduced, str):
        return None

    if not isinstance(reduced, tuple):
        msg = "Object __reduce_ex__ must return a string or tuple."
        raise TypeError(msg)

    if len(reduced) < 2 or len(reduced) > 6:
        msg = "Object __reduce_ex__ tuple must have between 2 and 6 items."
        raise TypeError(msg)

    callable_obj = cast("Callable[..., Any]", reduced[0])
    callable_args = cast("tuple[Any, ...]", reduced[1])
    if not callable(callable_obj):
        msg = "Object __reduce_ex__ tuple first item must be callable."
        raise TypeError(msg)

    if not isinstance(callable_args, tuple):
        msg = "Object __reduce_ex__ tuple second item must be a tuple."
        raise TypeError(msg)

    state = reduced[2] if len(reduced) > 2 else None
    list_items = reduced[3] if len(reduced) > 3 else None
    dict_items = reduced[4] if len(reduced) > 4 else None
    state_setter = cast("_StateSetter | None", reduced[5] if len(reduced) > 5 else None)
    if state_setter is not None and not callable(state_setter):
        msg = "Object __reduce_ex__ tuple sixth item must be callable or None."
        raise TypeError(msg)

    return callable_obj, callable_args, state, list_items, dict_items, state_setter


def _apply_mapping_state(instance: object, state: dict[Any, Any]) -> None:
    """Apply a dictionary-like state to an object."""
    instance_dict = getattr(instance, "__dict__", None)
    if isinstance(instance_dict, dict):
        instance_dict.update(state)
        return

    for key, value in state.items():
        if isinstance(key, str):
            setattr(instance, key, value)


def _apply_default_state(instance: object, state: object) -> None:
    """Apply state the same way pickle does without a custom state-setter."""
    if state is None:
        return

    instance_setstate = getattr(instance, "__setstate__", None)
    if callable(instance_setstate):
        instance_setstate(state)
        return

    if isinstance(state, dict):
        _apply_mapping_state(instance, state)
        return

    if isinstance(state, tuple) and len(state) == 2:
        dict_state, slot_state = state
        if isinstance(dict_state, dict):
            _apply_mapping_state(instance, dict_state)
        if isinstance(slot_state, dict):
            _apply_mapping_state(instance, slot_state)
        return

    msg = f"Unsupported pickle state without __setstate__: {type(state)}"
    raise TypeError(msg)


def _registry_state_setter(instance: object, state: object) -> None:
    """Restore regular pickle state and explicit RegistryMeta registrations."""
    if not isinstance(state, tuple) or len(state) != 4 or state[0] != _REGISTRY_STATE_MARKER:
        msg = "Invalid state payload for registry pickle restoration."
        raise RuntimeError(msg)

    original_state = state[1]
    registry_classes = state[2]
    original_state_setter = state[3]

    if callable(original_state_setter):
        state_setter = cast("_StateSetter", original_state_setter)
        state_setter(instance, original_state)
    else:
        _apply_default_state(instance, original_state)

    if isinstance(registry_classes, tuple):
        for registry_class in registry_classes:
            if isinstance(registry_class, RegistryMeta):
                cast("RegistryMeta", registry_class)._register_instance(instance)  # noqa: SLF001


class RegistryPickler(pickle.Pickler):
    """Pickler that preserves explicit RegistryMeta instance registrations."""

    def __init__(
        self,
        file: Any,  # noqa: ANN401
        protocol: int | None = None,
        *,
        fix_imports: bool = True,
        buffer_callback: Callable[[Any], object] | None = None,
    ) -> None:
        """Create a pickler that tracks explicit registry instance registrations."""
        super().__init__(
            file,
            protocol=protocol,
            fix_imports=fix_imports,
            buffer_callback=buffer_callback,
        )
        if protocol is None:
            self._protocol = pickle.DEFAULT_PROTOCOL
        elif protocol < 0:
            self._protocol = pickle.HIGHEST_PROTOCOL
        else:
            self._protocol = protocol

    def reducer_override(self, obj: object, /) -> Any:  # noqa: ANN401
        """Wrap reducer state for explicitly registered instances."""
        registered_classes = _collect_explicit_registry_classes(obj)
        if len(registered_classes) == 0:
            return NotImplemented

        normalized = _normalize_reduce(obj.__reduce_ex__(self._protocol))
        if normalized is None:
            return NotImplemented

        callable_obj, callable_args, state, list_items, dict_items, state_setter = normalized
        wrapped_state = (_REGISTRY_STATE_MARKER, state, tuple(registered_classes), state_setter)
        return (
            callable_obj,
            callable_args,
            wrapped_state,
            list_items,
            dict_items,
            _registry_state_setter,
        )


# `torch.save`/`torch.load` expects a pickle-like module with these attributes.
Pickler = RegistryPickler
Unpickler = pickle.Unpickler
DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL
HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def dump(
    obj: object,
    file: Any,  # noqa: ANN401
    protocol: int | None = None,
    *,
    fix_imports: bool = True,
    buffer_callback: Callable[[Any], object] | None = None,
) -> None:
    """Serialize an object to a file while preserving explicit registry registrations."""
    pickler = RegistryPickler(
        file,
        protocol=protocol,
        fix_imports=fix_imports,
        buffer_callback=buffer_callback,
    )
    pickler.dump(obj)


def dumps(
    obj: object,
    protocol: int | None = None,
    *,
    fix_imports: bool = True,
    buffer_callback: Callable[[Any], object] | None = None,
) -> bytes:
    """Serialize an object to bytes while preserving explicit registry registrations."""
    with io.BytesIO() as buffer:
        dump(
            obj,
            buffer,
            protocol=protocol,
            fix_imports=fix_imports,
            buffer_callback=buffer_callback,
        )
        return buffer.getvalue()


def load(
    file: Any,  # noqa: ANN401
    *,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
    buffers: Iterable[Any] | None = None,
) -> Any:  # noqa: ANN401
    """Deserialize an object previously written by dump or pickle.dump."""
    return pickle.load(  # noqa: S301
        file,
        fix_imports=fix_imports,
        encoding=encoding,
        errors=errors,
        buffers=buffers,
    )


def loads(
    data: bytes | bytearray | memoryview,
    *,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
    buffers: Iterable[Any] | None = None,
) -> Any:  # noqa: ANN401
    """Deserialize an object previously written by dumps or pickle.dumps."""
    return pickle.loads(  # noqa: S301
        data,
        fix_imports=fix_imports,
        encoding=encoding,
        errors=errors,
        buffers=buffers,
    )
